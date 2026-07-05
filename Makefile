# llamadeck — build / install / release helpers.
# The Go module lives in predictor/; these targets drive it from the repo root.

BINARY     := llamadeck
MODULE_DIR := predictor
PREFIX     ?= $(HOME)/.local
GO         ?= go
VERSION    := $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS    := -s -w -X main.version=$(VERSION)

.PHONY: build install uninstall test vet fmt snapshot release demo clean

build: ## Build the binary into predictor/llamadeck
	cd $(MODULE_DIR) && $(GO) build -ldflags "$(LDFLAGS)" -o $(BINARY) ./cmd/llamadeck
	@echo "built $(MODULE_DIR)/$(BINARY) ($(VERSION))"

install: build ## Install to $(PREFIX)/bin (default ~/.local/bin)
	mkdir -p $(PREFIX)/bin
	cp $(MODULE_DIR)/$(BINARY) $(PREFIX)/bin/$(BINARY)
	chmod +x $(PREFIX)/bin/$(BINARY)
	@echo "installed $(PREFIX)/bin/$(BINARY) — ensure $(PREFIX)/bin is on your PATH"

uninstall: ## Remove the installed binary
	rm -f $(PREFIX)/bin/$(BINARY)

test: ## Run the Go test suite
	cd $(MODULE_DIR) && $(GO) test ./...

vet: ## go vet
	cd $(MODULE_DIR) && $(GO) vet ./...

fmt: ## gofmt the module
	cd $(MODULE_DIR) && $(GO) fmt ./...

snapshot: ## Cross-platform build (no publish) via goreleaser
	goreleaser build --snapshot --clean

release: ## Tag-driven release via goreleaser (needs GITHUB_TOKEN)
	goreleaser release --clean

demo: build ## Render the README demo GIF (browser-free: needs python3 + agg)
	PATH="$(PWD)/$(MODULE_DIR):$(PATH)" python3 demo/gen_cast.py demo/demo.cast
	agg demo/demo.cast demo/llamadeck.gif --font-family "DejaVu Sans Mono" --font-size 20 --theme dracula
	@echo "wrote demo/llamadeck.gif"

demo-vhs: build ## Alternative GIF via vhs (needs vhs + ttyd + a display)
	PATH="$(PWD)/$(MODULE_DIR):$(PATH)" vhs demo/demo.tape

clean:
	rm -f $(MODULE_DIR)/$(BINARY)
	rm -rf dist
