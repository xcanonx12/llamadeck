package infra

import (
	"fmt"
	"net"
	"testing"
)

func TestSlugifyName(t *testing.T) {
	got := SlugifyName("unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M")
	want := "llama-unsloth-llama-3-2-1b-instruct-gguf-q4-k-m"
	if got != want {
		t.Errorf("SlugifyName = %q, want %q", got, want)
	}
}

func TestFreePortSkipsOccupied(t *testing.T) {
	// Occupy a port, then ensure FreePort doesn't hand it back.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Skip("cannot bind a port in this environment")
	}
	defer ln.Close()
	occupied := ln.Addr().(*net.TCPAddr).Port

	got := FreePort(occupied)
	if got == occupied {
		t.Fatalf("FreePort returned the occupied port %d", occupied)
	}
	// The returned port must actually be bindable.
	l2, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", got))
	if err != nil {
		t.Errorf("FreePort returned %d but it is not bindable: %v", got, err)
	} else {
		l2.Close()
	}
}
