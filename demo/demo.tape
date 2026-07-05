# VHS tape for the README demo GIF.  Record with:  make demo   (needs `vhs`)
# https://github.com/charmbracelet/vhs
Output demo/llamadeck.gif

Set Shell bash
Set FontSize 16
Set Width 1200
Set Height 780
Set Padding 18
Set TypingSpeed 35ms

Hide
Type "clear" Enter
Show

# --- the hook: which quant fits, before downloading anything ---
Type "# llamadeck: will this model run on my GPU — before you download it?"
Enter
Sleep 1500ms
Type "llamadeck recommend unsloth/Llama-3.2-1B-Instruct-GGUF --vram-mb 2000 --ram-mb 16000"
Enter
Sleep 6s

# --- predict a specific quant + context ---
Type "# check one quant at a chosen context"
Enter
Sleep 800ms
Type "llamadeck bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M --ctx 32768 --vram-mb 8000 --ram-mb 32000"
Enter
Sleep 4s

# --- the full control center ---
Type "# …or open the full control center"
Enter
Sleep 800ms
Type "llamadeck"
Enter
Sleep 3s
Type "2"
Sleep 2500ms
Type "1"
Sleep 2000ms
Type "q"
Sleep 1s
