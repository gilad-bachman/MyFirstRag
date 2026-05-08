"use client";

import { FormEvent, KeyboardEvent, useState } from "react";

type Status = "idle" | "loading" | "error";

function splitResponse(text: string): { body: string; sources: string | null } {
  const marker = "**Sources:**";
  const idx = text.lastIndexOf(marker);
  if (idx === -1) return { body: text, sources: null };
  return {
    body: text.slice(0, idx).trim(),
    sources: text.slice(idx + marker.length).trim(),
  };
}

export default function Page() {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [response, setResponse] = useState<string | null>(null);

  async function submit(text: string) {
    const trimmed = text.trim();
    if (!trimmed || status === "loading") return;
    setStatus("loading");
    setResponse(null);
    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ query: trimmed }),
      });
      if (!res.ok) throw new Error(String(res.status));
      const data = (await res.json()) as { response?: string; error?: string };
      if (data.error) throw new Error(data.error);
      setResponse(data.response ?? "");
      setStatus("idle");
    } catch {
      setStatus("error");
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    submit(query);
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit(query);
    }
  }

  const parsed = response ? splitResponse(response) : null;

  return (
    <main className="mx-auto flex min-h-screen max-w-2xl flex-col px-6 py-24">
      <header className="mb-16">
        <h1 className="font-serif text-4xl tracking-tight lowercase">bachrag</h1>
        <div className="mt-4 h-px w-full bg-[var(--color-rule)]" />
        <p className="mt-4 text-sm text-[var(--color-muted)]">
          a quiet research agent for gilad bachman.
        </p>
      </header>

      <form onSubmit={onSubmit} className="mb-12">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={onKeyDown}
          rows={2}
          placeholder="ask something…"
          disabled={status === "loading"}
          className="w-full resize-none border-0 border-b border-[var(--color-rule)] bg-transparent py-4 text-base text-[var(--color-ink)] placeholder:text-[var(--color-muted)] focus:border-[var(--color-ink)] focus:outline-none disabled:opacity-50"
        />
        <div className="mt-2 text-xs text-[var(--color-muted)]">
          enter to send · shift + enter for newline
        </div>
      </form>

      <section className="min-h-[8rem]">
        {status === "loading" && (
          <p className="italic text-[var(--color-muted)]">scanning documentation…</p>
        )}
        {status === "error" && (
          <p className="text-[#a14444]">connection lost.</p>
        )}
        {status === "idle" && parsed && (
          <article className="border-t border-[var(--color-rule)] pt-6">
            <div className="whitespace-pre-wrap leading-relaxed">{parsed.body}</div>
            {parsed.sources && (
              <div className="mt-6 text-xs text-[var(--color-muted)]">
                sources · {parsed.sources}
              </div>
            )}
          </article>
        )}
      </section>
    </main>
  );
}
