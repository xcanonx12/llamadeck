package app

// fitSettings is the inline, always-visible launch-settings panel on the Fit
// tab. Context / U-batch / KV-cache-type write straight into shared.cfg so the
// VRAM/RAM graph above recomputes live; the rest are launch-only knobs. There is
// no separate submenu and no modal — everything sits under the graph.

import (
	"fmt"
	"strings"

	"llamadeck/fit"
	"llamadeck/infra"
)

var kvTypes = []string{"f16", "q8_0", "q4_0"}

// fit settings row indices.
const (
	rCtx = iota
	rUBatch
	rKV
	rGPU
	rNGL
	rPort
	rThreads
	rParallel
	rMoE
	rFlash
	rJinja
	rNoMmap
	rMlock
	rowCount
)

type fitSettings struct {
	sh      *shared
	sel     int
	top     int // scroll offset: index of the first visible row
	editing bool
	editBuf string // digits typed in edit mode (empty ⇒ keep current on commit)
	device  int    // -1 = all GPUs; else a single GPU index
	nglAuto bool   // ngl follows the predicted recommendation
	ngl     int
	port    int
	threads int
	moe     int
	flash   bool
	jinja   bool
}

// numericRow reports whether a row holds a plain number that "e" can edit
// directly (booleans and the KV/GPU pickers are excluded on purpose).
func numericRow(row int) bool {
	switch row {
	case rCtx, rUBatch, rNGL, rPort, rThreads, rParallel, rMoE:
		return true
	}
	return false
}

// lockedInCPU reports whether a row is disabled in CPU mode (GPU-dependent, or
// KV which requires flash-attn we don't enable on CPU).
func lockedInCPU(row int) bool {
	switch row {
	case rGPU, rNGL, rMoE, rKV, rFlash:
		return true
	}
	return false
}

// moeSupported reports whether the selected model has experts to offload.
func (s *fitSettings) moeSupported() bool {
	return s.sh.selected != nil && s.sh.selected.model != nil && s.sh.selected.model.IsMoE()
}

// startEdit enters direct-entry mode for the selected numeric row.
func (s *fitSettings) startEdit() {
	if !numericRow(s.sel) {
		return
	}
	if s.sh.cpuMode && lockedInCPU(s.sel) {
		return
	}
	if s.sel == rMoE && !s.moeSupported() {
		return
	}
	s.editing, s.editBuf = true, ""
}

// editKey handles a keypress while editing a numeric field.
func (s *fitSettings) editKey(key string) {
	switch {
	case key == "enter":
		s.commitEdit()
	case key == "esc":
		s.editing, s.editBuf = false, ""
	case key == "backspace":
		if n := len(s.editBuf); n > 0 {
			s.editBuf = s.editBuf[:n-1]
		}
	case len(key) == 1 && key[0] >= '0' && key[0] <= '9':
		if len(s.editBuf) < 9 { // guard against overflow
			s.editBuf += key
		}
	}
}

// commitEdit parses the buffer and writes it to the selected field.
func (s *fitSettings) commitEdit() {
	s.editing = true // stays until we clear below (keeps focus during commit)
	buf := s.editBuf
	s.editing, s.editBuf = false, ""
	if buf == "" {
		return
	}
	v := 0
	for i := 0; i < len(buf); i++ {
		v = v*10 + int(buf[i]-'0')
	}
	s.applyNumeric(s.sel, v)
}

// applyNumeric writes a value to a numeric row, clamping where it matters.
func (s *fitSettings) applyNumeric(row, v int) {
	switch row {
	case rCtx:
		if v < fitMinCtx {
			v = fitMinCtx
		}
		if v > fitMaxCtx {
			v = fitMaxCtx
		}
		s.sh.cfg.Ctx = v
	case rUBatch:
		if v < fitMinUBatch {
			v = fitMinUBatch
		}
		if v > fitMaxUBatch {
			v = fitMaxUBatch
		}
		s.sh.cfg.UBatch = v
	case rNGL:
		s.nglAuto, s.ngl = false, v
	case rPort:
		s.port = v
	case rThreads:
		s.threads = v
	case rParallel:
		s.sh.cfg.NSeqs = v // shared cfg: the graph's RS estimate reacts live
	case rMoE:
		if s.sh.selected != nil && s.sh.selected.model != nil && v > s.sh.selected.model.NLayers {
			v = s.sh.selected.model.NLayers
		}
		s.moe = v
	}
}

func newFitSettings(sh *shared, port int) *fitSettings {
	return &fitSettings{sh: sh, port: port, device: -1, nglAuto: true, flash: true}
}

// gpuCount is the number of GPUs available for pinning.
func (s *fitSettings) gpuCount() int {
	if n := len(s.sh.gpuFree); n > 0 {
		return n
	}
	return s.sh.hw.NumGPUs
}

// hw returns the hardware to predict against: a single pinned device's free
// VRAM (so the estimate matches what that GPU can actually hold), or the pooled
// default when targeting all GPUs.
func (s *fitSettings) hw() fit.Hardware {
	if s.sh.cpuMode {
		return fit.Hardware{FreeRAM: s.sh.hw.FreeRAM}
	}
	if s.device >= 0 && s.device < len(s.sh.gpuFree) {
		return fit.Hardware{FreeVRAM: s.sh.gpuFree[s.device], FreeRAM: s.sh.hw.FreeRAM, NumGPUs: 1}
	}
	// Targeting all GPUs: pass per-device free VRAM so the graph can show the
	// per-device split (pooled Predict ignores GPUsFree, so this stays back-compat).
	hw := s.sh.hw
	if len(s.sh.gpuFree) > 1 {
		hw.GPUsFree = s.sh.gpuFree
	}
	return hw
}

func (s *fitSettings) up() {
	if s.sel > 0 {
		s.sel--
	}
}
func (s *fitSettings) down() {
	if s.sel < rowCount-1 {
		s.sel++
	}
}

func clampMul(v, factor, min, max int) int {
	v *= factor
	if v < min {
		v = min
	}
	if v > max {
		v = max
	}
	return v
}

// adjust mutates the selected field for a key (←→/space/digits/backspace).
func (s *fitSettings) adjust(key string) {
	if s.sh.cpuMode && lockedInCPU(s.sel) {
		return
	}
	digit := len(key) == 1 && key[0] >= '0' && key[0] <= '9'
	switch s.sel {
	case rCtx:
		if key == "left" {
			s.sh.cfg.Ctx = clampMul(s.sh.cfg.Ctx, 1, fitMinCtx, fitMaxCtx) / 2
			if s.sh.cfg.Ctx < fitMinCtx {
				s.sh.cfg.Ctx = fitMinCtx
			}
		} else if key == "right" {
			s.sh.cfg.Ctx = clampMul(s.sh.cfg.Ctx, 2, fitMinCtx, fitMaxCtx)
		}
	case rUBatch:
		if key == "left" {
			s.sh.cfg.UBatch /= 2
			if s.sh.cfg.UBatch < fitMinUBatch {
				s.sh.cfg.UBatch = fitMinUBatch
			}
		} else if key == "right" {
			s.sh.cfg.UBatch = clampMul(s.sh.cfg.UBatch, 2, fitMinUBatch, fitMaxUBatch)
		}
	case rKV:
		i := kvIndex(s.sh.cfg.KVType)
		switch key {
		case "right", " ":
			i = (i + 1) % len(kvTypes)
		case "left":
			i = (i - 1 + len(kvTypes)) % len(kvTypes)
		}
		s.sh.cfg.KVType = kvTypes[i]
	case rGPU:
		n := s.gpuCount()
		switch key {
		case "right", " ":
			if s.device+1 >= n {
				s.device = -1
			} else {
				s.device++
			}
		case "left":
			if s.device <= -1 {
				s.device = n - 1
			} else {
				s.device--
			}
		}
	case rNGL:
		switch {
		case key == "left":
			if s.nglAuto {
				// stay auto
			} else if s.ngl <= 0 {
				s.nglAuto = true
			} else {
				s.ngl--
			}
		case key == "right":
			if s.nglAuto {
				s.nglAuto, s.ngl = false, 0
			} else {
				s.ngl++
			}
		case digit:
			s.nglAuto = false
			s.ngl = capInt(s.ngl*10 + int(key[0]-'0'))
		case key == "backspace":
			s.ngl /= 10
		}
	case rPort:
		s.port = editInt(s.port, key, digit)
	case rThreads:
		s.threads = editInt(s.threads, key, digit)
	case rParallel:
		s.sh.cfg.NSeqs = editInt(s.sh.cfg.NSeqs, key, digit)
	case rMoE:
		if !s.moeSupported() {
			return // dense model — nothing to offload
		}
		s.moe = editInt(s.moe, key, digit)
		if s.sh.selected != nil && s.sh.selected.model != nil && s.moe > s.sh.selected.model.NLayers {
			s.moe = s.sh.selected.model.NLayers
		}
	case rFlash:
		toggle(&s.flash, key)
	case rJinja:
		toggle(&s.jinja, key)
	case rNoMmap:
		toggle(&s.sh.cfg.NoMmap, key) // shared cfg: flips the RAM verdict (paged vs hard)
	case rMlock:
		toggle(&s.sh.cfg.Mlock, key)
	}
}

func kvIndex(t string) int {
	for i, v := range kvTypes {
		if v == t {
			return i
		}
	}
	return 0
}

func capInt(v int) int {
	if v > 1_000_000 {
		return 1_000_000
	}
	return v
}

func editInt(v int, key string, digit bool) int {
	switch {
	case digit:
		return capInt(v*10 + int(key[0]-'0'))
	case key == "backspace":
		return v / 10
	case key == "left" && v > 0:
		return v - 1
	case key == "right":
		return v + 1
	}
	return v
}

func toggle(b *bool, key string) {
	if key == "left" || key == "right" || key == " " {
		*b = !*b
	}
}

// effNGL returns the -ngl to launch with. Auto = 999: an over-large -ngl makes
// llama.cpp auto-fit (llama_params_fit load-balances layers — the split the
// auto graph models). Passing the suggested COUNT explicitly instead would
// trigger llama's obey-it-exactly proportional split — a different placement
// that can OOM the main GPU even though the auto graph said "fits".
func (s *fitSettings) effNGL() int {
	if s.nglAuto {
		return 999
	}
	return s.ngl
}

// toSpec builds the launch spec. KV-cache type is taken from cfg (shared with
// the graph). Quantized KV requires flash-attention in llama.cpp, so we force it
// on in that case to avoid a launch-time failure.
func (s *fitSettings) toSpec(hf, cacheDir string) infra.LaunchSpec {
	if s.sh.cpuMode {
		return infra.LaunchSpec{
			HF: hf, Port: s.port, Ctx: s.sh.cfg.Ctx, NGL: 0, CacheDir: cacheDir,
			UBatch: s.sh.cfg.UBatch, Threads: s.threads, Parallel: s.sh.cfg.NSeqs,
			Jinja: s.jinja, NoMmap: s.sh.cfg.NoMmap, Mlock: s.sh.cfg.Mlock,
			CPU: true,
		}
	}
	cache := s.sh.cfg.KVType
	if cache == "f16" {
		cache = "" // server default — omit the flag
	}
	flash := s.flash
	if cache != "" {
		flash = true
	}
	device := ""
	if s.device >= 0 {
		device = fmt.Sprint(s.device)
	}
	moe := s.moe
	if !s.moeSupported() {
		moe = 0 // never pass --n-cpu-moe to a dense model
	}
	return infra.LaunchSpec{
		HF: hf, Port: s.port, Ctx: s.sh.cfg.Ctx, NGL: s.effNGL(),
		CacheDir: cacheDir, UBatch: s.sh.cfg.UBatch, FlashAttn: flash, Device: device,
		Threads: s.threads, NCPUMoE: moe, Parallel: s.sh.cfg.NSeqs,
		CacheTypeK: cache, CacheTypeV: cache,
		Jinja: s.jinja, NoMmap: s.sh.cfg.NoMmap, Mlock: s.sh.cfg.Mlock,
	}
}

func onOff(b bool) string {
	if b {
		return "on"
	}
	return "off"
}

func intOr(v int, zero string) string {
	if v == 0 {
		return zero
	}
	return fmt.Sprint(v)
}

// render draws the settings list, windowed to maxRows so it never overflows the
// terminal (the graph above stays fixed). suggested is the predicted ngl;
// safeMax is the largest explicit -ngl predicted to survive load (0 = unknown
// or not applicable), used to warn on the -ngl row.
func (s *fitSettings) render(suggested, safeMax, maxRows int) string {
	moeVal := intOr(s.moe, "none")
	if !s.moeSupported() {
		moeVal = "n/a (dense model)"
	}
	rows := []struct{ label, val string }{
		rCtx:      {"Context", fmt.Sprint(s.sh.cfg.Ctx)},
		rUBatch:   {"U-batch", fmt.Sprint(s.sh.cfg.UBatch)},
		rKV:       {"KV cache type", s.sh.cfg.KVType},
		rGPU:      {"GPU target", s.deviceLabel()},
		rNGL:      {"GPU layers (-ngl)", nglLabel(s.nglAuto, s.ngl, suggested, safeMax)},
		rPort:     {"Port", intOr(s.port, "auto (next free ≥8080)")},
		rThreads:  {"Threads", intOr(s.threads, "auto")},
		rParallel: {"Parallel slots", intOr(s.sh.cfg.NSeqs, "auto (4)")},
		rMoE:      {"MoE layers→CPU", moeVal},
		rFlash:    {"Flash attention", onOff(s.flash)},
		rJinja:    {"Jinja template", onOff(s.jinja)},
		rNoMmap:   {"No mmap", onOff(s.sh.cfg.NoMmap)},
		rMlock:    {"mlock (pin RAM)", onOff(s.sh.cfg.Mlock)},
	}
	if s.sh.cpuMode {
		rows[rGPU].val = "n/a (CPU)"
		rows[rNGL].val = "n/a (CPU)"
		rows[rMoE].val = "n/a (CPU)"
		rows[rKV].val = "f16 (CPU)"
		rows[rFlash].val = "n/a (CPU)"
	}
	// While editing, show the live buffer with a cursor on the selected row.
	if s.editing {
		rows[s.sel].val = s.editBuf + "▌"
	}

	// Keep the selection inside the visible window. When the list doesn't all fit,
	// reserve 2 lines for the "N more" markers so the block never exceeds maxRows.
	if maxRows < 1 {
		maxRows = 1
	}
	visN := len(rows)
	if visN > maxRows {
		visN = maxRows - 2
		if visN < 1 {
			visN = 1
		}
	}
	if s.sel < s.top {
		s.top = s.sel
	}
	if s.sel >= s.top+visN {
		s.top = s.sel - visN + 1
	}
	if s.top < 0 {
		s.top = 0
	}
	end := s.top + visN
	if end > len(rows) {
		end = len(rows)
	}

	var b strings.Builder
	if s.top > 0 {
		b.WriteString(stMuted.Render(fmt.Sprintf("  ↑ %d more", s.top)) + "\n")
	}
	for i := s.top; i < end; i++ {
		r := rows[i]
		if i == s.sel {
			b.WriteString(stSelected.Render("› "+pad(pad(r.label, 20)+" "+r.val, 34)) + "\n")
		} else {
			b.WriteString("  " + stMuted.Render(pad(r.label, 20)) + " " + stText.Render(r.val) + "\n")
		}
	}
	if end < len(rows) {
		b.WriteString(stMuted.Render(fmt.Sprintf("  ↓ %d more", len(rows)-end)))
	}
	return strings.TrimRight(b.String(), "\n")
}

func nglLabel(auto bool, ngl, suggested, safeMax int) string {
	if auto {
		return fmt.Sprintf("auto (%d)", suggested)
	}
	// Warn when the explicit choice exceeds the largest -ngl predicted to
	// survive load (the load-time margin, not just the paper fit).
	if safeMax > 0 && ngl > safeMax {
		return fmt.Sprintf("%d ⚠ (max safe %d)", ngl, safeMax)
	}
	return fmt.Sprint(ngl)
}

// deviceLabel describes the GPU target, with that device's free VRAM when pinned.
func (s *fitSettings) deviceLabel() string {
	if s.device < 0 {
		return fmt.Sprintf("all (%d GPUs)", max(1, s.gpuCount()))
	}
	free := ""
	if s.device < len(s.sh.gpuFree) {
		free = " · " + human(s.sh.gpuFree[s.device]) + " free"
	}
	return fmt.Sprintf("GPU %d%s", s.device, free)
}
