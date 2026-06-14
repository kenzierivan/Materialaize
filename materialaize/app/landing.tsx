"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { createPortal } from "react-dom";

// ─── Constants ────────────────────────────────────────────────────────────────

const ACCENT = "#00ff9d";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const PHASES = [
  "ENCODING_SELFIES",
  "AUTOREGRESSIVE_SAMPLING",
  "DECODING_SMILES",
  "RDKIT_VALIDATION",
  "MOTIF_SCORING",
];

type Molecule = {
  selfies: string;
  smiles: string | null;
  valid: boolean;
  canonical_smiles?: string;
  is_polymer?: boolean;
  molecular_weight?: number;
  num_heavy_atoms?: number;
  num_rings?: number;
  motifs_found?: string[];
  has_any_sh_motif?: boolean;
  has_strict_sh?: boolean;
  sa_score?: number | null;
  synthesisable?: boolean | null;
};

// Backend may return either a number or the literal string "N/A" for any
// metric that depends on optional state (e.g. SA scorer not loaded, < 2 valid
// molecules for diversity). Using a permissive number|string union.
type BatchMetrics = {
  "validity_%": number;
  n_valid: number;
  n_total: number;
  "continuation_polymer_%": number;
  "uniqueness_%": number;
  diversity: number | string;
  "self_healing_motif_%": number;
  "strict_sh_%": number;
  mean_sa_score: number | string;
  "synthesisable_%": number | string;
  [perMotif: string]: number | string | null | undefined;
};

type Candidate = Molecule & { id: string; shScore: number };

// ─── API ──────────────────────────────────────────────────────────────────────

type GenerateResponse = {
  seed: string;
  molecules: Molecule[];
  batch_metrics: BatchMetrics;
};

async function generateMolecules(
  seed: string,
  numMolecules: number,
): Promise<{ molecules: Molecule[]; batchMetrics: BatchMetrics }> {
  const res = await fetch(`${API_URL}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      seed,
      num_molecules: numMolecules,
      max_new_tokens: 80,
      temperature: 0.8,
      top_k: 50,
      top_p: 0.95,
    }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`Backend ${res.status}: ${text || res.statusText}`);
  }
  const data = (await res.json()) as GenerateResponse;
  return { molecules: data.molecules, batchMetrics: data.batch_metrics };
}

async function explainMolecule(c: Candidate): Promise<string> {
  const res = await fetch(`${API_URL}/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      smiles: c.canonical_smiles ?? c.smiles ?? "",
      motifs_found: c.motifs_found ?? [],
      sa_score: c.sa_score ?? null,
      molecular_weight: c.molecular_weight ?? null,
      num_heavy_atoms: c.num_heavy_atoms ?? null,
      num_rings: c.num_rings ?? null,
      is_polymer: c.is_polymer ?? null,
      synthesisable: c.synthesisable ?? null,
      has_strict_sh: c.has_strict_sh ?? null,
    }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`Backend ${res.status}: ${text || res.statusText}`);
  }
  const data = (await res.json()) as { explanation: string };
  return data.explanation;
}

// Deterministic FNV hash → MTZ-XXXX card label.
function makeId(seed: string): string {
  let h = 2166136261;
  for (let i = 0; i < seed.length; i++) {
    h ^= seed.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  const hex = ((h >>> 0).toString(16) + "00000000").slice(0, 8).toUpperCase();
  return `MTZ-${hex.slice(0, 4)}`;
}

// Composite "Self-Healing Quality" — combines real backend signals into one
// 0-100 number per candidate so cards can be compared at a glance:
//   +45 strict SH motif (disulfide/urea — real covalent-exchange chemistries)
//   +20 any other SH motif (amide H-bond, furan, imine, boronic ester, DA)
//   +20 polymer continuation (* atoms — model thinks this extends as a chain)
//   +20 synthesisable (SA score ≤ 6)
//   +15 MW in monomer/oligomer range (100–1000 g/mol)
//   +5 per motif beyond the first (capped at +15) — rewards multi-mechanism
//      candidates that combine e.g. disulfide + urea + amide for more robust
//      self-healing than a single-motif molecule
function selfHealingScore(m: Molecule): number {
  if (!m.valid) return 0;
  let s = 0;
  if (m.has_strict_sh) s += 45;
  else if (m.has_any_sh_motif) s += 20;
  if (m.is_polymer) s += 20;
  if (m.synthesisable === true) s += 20;
  const mw = m.molecular_weight ?? 0;
  if (mw >= 100 && mw <= 1000) s += 15;
  const motifCount = m.motifs_found?.length ?? 0;
  s += Math.min(15, Math.max(0, motifCount - 1) * 5);
  return Math.min(100, s);
}

// Same rules as selfHealingScore, but returns each contribution separately
// for the EXPAND DETAILS panel. Keep these two functions in sync.
type SHContribution = {
  label: string;
  awarded: number;
  max: number;
  reason: string;
};

function selfHealingBreakdown(m: Molecule): SHContribution[] {
  const mw = m.molecular_weight ?? 0;
  const inMwRange = mw >= 100 && mw <= 1000;
  const motifCount = m.motifs_found?.length ?? 0;
  const motifBonus = Math.min(15, Math.max(0, motifCount - 1) * 5);
  return [
    {
      label: "strict SH motif",
      awarded: m.has_strict_sh ? 45 : 0,
      max: 45,
      reason: "disulfide or urea (covalent-exchange chemistry)",
    },
    {
      label: "any SH motif",
      awarded: !m.has_strict_sh && m.has_any_sh_motif ? 20 : 0,
      max: 20,
      reason: "amide H-bond / furan / imine / boronic ester / DA",
    },
    {
      label: "polymer continuation",
      awarded: m.is_polymer ? 20 : 0,
      max: 20,
      reason: "* marker in canonical SMILES",
    },
    {
      label: "synthesisable",
      awarded: m.synthesisable === true ? 20 : 0,
      max: 20,
      reason: "SA score ≤ 6",
    },
    {
      label: "MW range",
      awarded: inMwRange ? 15 : 0,
      max: 15,
      reason: "100–1000 g/mol (monomer/oligomer)",
    },
    {
      label: "motif diversity",
      awarded: motifBonus,
      max: 15,
      reason: `+5 per motif beyond the first (found ${motifCount})`,
    },
  ];
}

function toCandidate(m: Molecule): Candidate {
  // Backend sends smiles=null for failed decodes — skip it for ID hashing
  // (otherwise multiple invalid candidates collide on the same fallback).
  const idSeed = m.canonical_smiles || m.selfies || "?";
  return { ...m, id: makeId(idSeed), shScore: selfHealingScore(m) };
}

// Format a metric that could be a number or the literal string "N/A".
function formatMetric(
  v: number | string | null | undefined,
  digits: number,
): string {
  if (typeof v === "number") return v.toFixed(digits);
  if (typeof v === "string") return v;
  return "N/A";
}

// ─── Lattice Canvas ───────────────────────────────────────────────────────────

function Lattice() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef(0);
  const tRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let W = 0,
      H = 0;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      W = rect.width;
      H = rect.height;
      canvas.width = Math.floor(W * dpr);
      canvas.height = Math.floor(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    // Build 5×5×5 lattice with organic jitter
    const N = 5;
    const spacing = 80;
    const nodes: {
      x: number;
      y: number;
      z: number;
      ix: number;
      iy: number;
      iz: number;
    }[] = [];
    for (let x = 0; x < N; x++) {
      for (let y = 0; y < N; y++) {
        for (let z = 0; z < N; z++) {
          nodes.push({
            x: (x - (N - 1) / 2) * spacing + (Math.random() - 0.5) * 14,
            y: (y - (N - 1) / 2) * spacing + (Math.random() - 0.5) * 14,
            z: (z - (N - 1) / 2) * spacing + (Math.random() - 0.5) * 14,
            ix: x,
            iy: y,
            iz: z,
          });
        }
      }
    }
    const edges: [number, number][] = [];
    const idx = (x: number, y: number, z: number) => x * N * N + y * N + z;
    for (let x = 0; x < N; x++) {
      for (let y = 0; y < N; y++) {
        for (let z = 0; z < N; z++) {
          if (x < N - 1) edges.push([idx(x, y, z), idx(x + 1, y, z)]);
          if (y < N - 1) edges.push([idx(x, y, z), idx(x, y + 1, z)]);
          if (z < N - 1) edges.push([idx(x, y, z), idx(x, y, z + 1)]);
        }
      }
    }

    // Parse accent hex to rgb once
    const m = ACCENT.replace("#", "");
    const [r, g, b] = [
      parseInt(m.slice(0, 2), 16),
      parseInt(m.slice(2, 4), 16),
      parseInt(m.slice(4, 6), 16),
    ];

    const project = (x: number, y: number, z: number) => {
      const focal = 700;
      const scale = focal / (focal + z);
      return { x: W / 2 + x * scale, y: H / 2 + y * scale, s: scale, z };
    };

    const draw = () => {
      tRef.current += 0.003;
      const t = tRef.current;
      ctx.clearRect(0, 0, W, H);

      const cosX = Math.cos(t * 0.4),
        sinX = Math.sin(t * 0.4);
      const cosY = Math.cos(t * 0.6),
        sinY = Math.sin(t * 0.6);

      const projected = nodes.map((n) => {
        const y1 = n.y * cosX - n.z * sinX;
        const z1 = n.y * sinX + n.z * cosX;
        const x2 = n.x * cosY + z1 * sinY;
        const z2 = -n.x * sinY + z1 * cosY;
        const breathe = Math.sin(t * 1.5 + n.ix + n.iy + n.iz) * 4;
        return project(x2, y1 + breathe, z2);
      });

      edges.forEach(([i, j]) => {
        const a = projected[i],
          c = projected[j];
        const depth = ((a.z + c.z) / 2 + 400) / 800;
        const alpha = Math.max(0.03, 0.45 * (1 - depth));
        ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
        ctx.lineWidth = 0.6;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(c.x, c.y);
        ctx.stroke();
      });

      projected.forEach((p) => {
        const depth = (p.z + 400) / 800;
        const alpha = Math.max(0.1, 1 - depth);
        const size = Math.max(0.8, p.s * 2.2);
        ctx.fillStyle = `rgba(${r},${g},${b},${alpha * 0.9})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
        ctx.fill();
        if (alpha > 0.5) {
          ctx.fillStyle = `rgba(${r},${g},${b},${alpha * 0.15})`;
          ctx.beginPath();
          ctx.arc(p.x, p.y, size * 3, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(rafRef.current);
      ro.disconnect();
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        inset: 0,
        width: "100%",
        height: "100%",
        display: "block",
      }}
    />
  );
}

// ─── Small helpers ────────────────────────────────────────────────────────────

function Logo() {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <svg width="22" height="22" viewBox="0 0 22 22">
        <circle cx="4" cy="11" r="2.4" fill={ACCENT} />
        <circle
          cx="11"
          cy="4"
          r="2.4"
          fill="none"
          stroke={ACCENT}
          strokeWidth="1.2"
        />
        <circle cx="18" cy="11" r="2.4" fill={ACCENT} />
        <circle
          cx="11"
          cy="18"
          r="2.4"
          fill="none"
          stroke={ACCENT}
          strokeWidth="1.2"
        />
        <line
          x1="4"
          y1="11"
          x2="11"
          y2="4"
          stroke={ACCENT}
          strokeWidth="1"
          opacity="0.5"
        />
        <line
          x1="11"
          y1="4"
          x2="18"
          y2="11"
          stroke={ACCENT}
          strokeWidth="1"
          opacity="0.5"
        />
        <line
          x1="18"
          y1="11"
          x2="11"
          y2="18"
          stroke={ACCENT}
          strokeWidth="1"
          opacity="0.5"
        />
        <line
          x1="11"
          y1="18"
          x2="4"
          y2="11"
          stroke={ACCENT}
          strokeWidth="1"
          opacity="0.5"
        />
      </svg>
      <span
        style={{
          fontFamily: "var(--mono)",
          fontSize: 13,
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          color: "var(--fg)",
        }}
      >
        MATERIAL<span style={{ color: ACCENT }}>AI</span>ZE
      </span>
    </div>
  );
}

function BlinkDot() {
  return (
    <span
      className="blink"
      style={{
        display: "inline-block",
        width: 6,
        height: 6,
        borderRadius: "50%",
        background: ACCENT,
        marginRight: 8,
        verticalAlign: 1,
      }}
    />
  );
}

function Bracket({
  pos,
  color,
}: {
  pos: "tl" | "tr" | "bl" | "br";
  color: string;
}) {
  const size = 8;
  const base: React.CSSProperties = {
    position: "absolute",
    width: size,
    height: size,
    pointerEvents: "none",
    transition: "border-color 0.25s",
  };
  const sides: Record<string, React.CSSProperties> = {
    tl: {
      top: -1,
      left: -1,
      borderTop: `1px solid ${color}`,
      borderLeft: `1px solid ${color}`,
    },
    tr: {
      top: -1,
      right: -1,
      borderTop: `1px solid ${color}`,
      borderRight: `1px solid ${color}`,
    },
    bl: {
      bottom: -1,
      left: -1,
      borderBottom: `1px solid ${color}`,
      borderLeft: `1px solid ${color}`,
    },
    br: {
      bottom: -1,
      right: -1,
      borderBottom: `1px solid ${color}`,
      borderRight: `1px solid ${color}`,
    },
  };
  return <span style={{ ...base, ...sides[pos] }} />;
}

// ─── Nav ──────────────────────────────────────────────────────────────────────

function Nav() {
  return (
    <nav
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "24px 40px",
        zIndex: 10,
      }}
    >
      <Logo />
      <div
        className="nav-links"
        style={{
          display: "flex",
          gap: 28,
          fontFamily: "var(--mono)",
          fontSize: 11,
          letterSpacing: "0.14em",
          textTransform: "uppercase",
          color: "var(--fg-dim)",
        }}
      >
        <a href="#how" style={{ color: "inherit", textDecoration: "none" }}>
          How it works
        </a>
        <a
          href="#research"
          style={{ color: "inherit", textDecoration: "none" }}
        >
          Research
        </a>
        <a
          href="#"
          style={{ color: "var(--fg)", textDecoration: "none" }}
        >
          Request access <span style={{ color: ACCENT }}>→</span>
        </a>
      </div>
    </nav>
  );
}

// ─── Hero Input ───────────────────────────────────────────────────────────────

function HeroInput({ onSubmit }: { onSubmit: (seed: string) => void }) {
  const [value, setValue] = useState("");
  const [focused, setFocused] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const submit = (e?: React.FormEvent) => {
    e?.preventDefault();
    // Default seed: a self-healing polymer (urea + disulfide motifs) with *
    // endpoints. Keep simple — backend rejects seeds that can't be SELFIES-encoded.
    onSubmit(value.trim() || "*NC(=O)NCCSSCC*");
  };

  const bracketColor = focused ? ACCENT : "var(--border-strong)";

  return (
    <form onSubmit={submit} style={{ width: "100%", maxWidth: 720 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontFamily: "var(--mono)",
          fontSize: 10,
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          color: "var(--fg-dim)",
          marginBottom: 10,
        }}
      >
        <span>
          <span style={{ color: ACCENT }}>›</span> SMILES_SEED
          <span style={{ opacity: 0.5 }}> · ENTER MOLECULAR INPUT</span>
        </span>
        <span>
          MODEL <span style={{ color: "var(--fg)" }}>v1.0</span> · 4.7M params
        </span>
      </div>

      <div
        style={{
          position: "relative",
          background: "rgba(8, 11, 13, 0.72)",
          backdropFilter: "blur(12px)",
          WebkitBackdropFilter: "blur(12px)",
          border: `1px solid ${focused ? ACCENT : "var(--border)"}`,
          boxShadow: "inset 0 1px 0 rgba(255,255,255,0.04)",
          transition: "border-color 0.25s",
        }}
      >
        <div
          style={{
            position: "absolute",
            inset: -1,
            boxShadow: `0 0 0 1px ${ACCENT}, 0 0 60px -20px ${ACCENT}`,
            opacity: focused ? 1 : 0,
            transition: "opacity 0.25s",
            pointerEvents: "none",
            zIndex: 0,
          }}
        />
        <Bracket pos="tl" color={bracketColor} />
        <Bracket pos="tr" color={bracketColor} />
        <Bracket pos="bl" color={bracketColor} />
        <Bracket pos="br" color={bracketColor} />

        <div style={{ display: "flex", alignItems: "stretch" }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              padding: "0 16px",
              color: ACCENT,
              fontFamily: "var(--mono)",
              fontSize: 14,
              borderRight: "1px solid var(--border)",
            }}
          >
            $
          </div>
          <input
            ref={inputRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            placeholder="*NC(=O)NCCSSCC*"
            spellCheck={false}
            autoComplete="off"
            style={{
              flex: 1,
              padding: "20px 16px",
              background: "transparent",
              border: "none",
              outline: "none",
              color: "var(--fg)",
              fontFamily: "var(--mono)",
              fontSize: 15,
              letterSpacing: "0.02em",
              caretColor: ACCENT,
            }}
          />
          <button
            type="submit"
            style={{
              padding: "0 24px",
              background: ACCENT,
              color: "#050607",
              border: "none",
              fontFamily: "var(--mono)",
              fontSize: 11,
              fontWeight: 600,
              letterSpacing: "0.18em",
              textTransform: "uppercase",
              cursor: "pointer",
              transition: "filter 0.2s",
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}
            onMouseEnter={(e) =>
              (e.currentTarget.style.filter = "brightness(1.15)")
            }
            onMouseLeave={(e) =>
              (e.currentTarget.style.filter = "brightness(1)")
            }
          >
            Generate
            <span style={{ fontSize: 14 }}>↗</span>
          </button>
        </div>
      </div>

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginTop: 12,
          fontFamily: "var(--mono)",
          fontSize: 10,
          letterSpacing: "0.16em",
          textTransform: "uppercase",
          color: "var(--fg-dim)",
        }}
      >
        <span>
          <BlinkDot /> READY · ChemGPT-4.7M + LoRA · RDKit motif scoring
        </span>
        <span>↵ TO GENERATE 3 CANDIDATES</span>
      </div>
    </form>
  );
}

// ─── Telemetry strip ──────────────────────────────────────────────────────────

type Health = {
  status?: string;
  model?: string;
  device?: string;
  selfies_version?: string;
  sa_scorer?: boolean;
};

function Telemetry() {
  const [health, setHealth] = useState<Health | null>(null);
  const [reachable, setReachable] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_URL}/health`)
      .then((r) => r.json())
      .then((j: Health) => {
        if (!cancelled) {
          setHealth(j);
          setReachable(true);
        }
      })
      .catch(() => {
        if (!cancelled) setReachable(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const dash = "—";
  const items = [
    { k: "BASE", v: "ChemGPT-4.7M" },
    { k: "ADAPTER", v: "LoRA · r32" },
    {
      k: "SELFIES",
      v: health?.selfies_version ? `v${health.selfies_version}` : dash,
    },
    {
      k: "DEVICE",
      v: reachable === false ? "offline" : (health?.device ?? dash),
    },
  ];

  return (
    <div
      style={{
        position: "absolute",
        bottom: 32,
        left: 40,
        right: 40,
        display: "flex",
        justifyContent: "space-between",
        fontFamily: "var(--mono)",
        fontSize: 10,
        letterSpacing: "0.16em",
        textTransform: "uppercase",
        color: "var(--fg-dim)",
        pointerEvents: "none",
        zIndex: 2,
      }}
    >
      {items.map(({ k, v }) => (
        <span key={k}>
          {k} <span style={{ color: ACCENT, marginLeft: 6 }}>{v}</span>
        </span>
      ))}
    </div>
  );
}

// ─── Hero ─────────────────────────────────────────────────────────────────────

function Hero({ onSubmit }: { onSubmit: (seed: string) => void }) {
  return (
    <section
      style={{
        position: "relative",
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "120px 40px 80px",
        overflow: "hidden",
      }}
    >
      {/* Vignette */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background:
            "radial-gradient(ellipse at center, transparent 0%, rgba(5,6,7,0.4) 50%, rgba(5,6,7,0.92) 100%)",
          pointerEvents: "none",
        }}
      />

      <div
        style={{
          position: "relative",
          zIndex: 2,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          width: "100%",
          maxWidth: 880,
          textAlign: "center",
        }}
      >
        {/* Status badge */}
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 10,
            padding: "6px 12px",
            border: "1px solid var(--border)",
            borderRadius: 999,
            fontFamily: "var(--mono)",
            fontSize: 10,
            letterSpacing: "0.18em",
            textTransform: "uppercase",
            color: "var(--fg-dim)",
            background: "rgba(8,11,13,0.6)",
            backdropFilter: "blur(8px)",
            marginBottom: 32,
          }}
        >
          <BlinkDot />
          ChemGPT-4.7M + LoRA · 7 self-healing motif checks
        </div>

        {/* Headline */}
        <h1
          style={{
            fontFamily: "var(--display)",
            fontSize: "clamp(44px, 6vw, 88px)",
            fontWeight: 400,
            lineHeight: 0.98,
            letterSpacing: "-0.03em",
            margin: 0,
            color: "var(--fg)",
          }}
        >
          Generate self-healing
          <br />
          <span style={{ whiteSpace: "nowrap" }}>
            polymers from a{" "}
            <span
              style={{
                fontStyle: "italic",
                fontWeight: 300,
                color: ACCENT,
                textShadow: `0 0 40px ${ACCENT}55`,
              }}
            >
              seed
            </span>
            .
          </span>
        </h1>

        <p
          style={{
            marginTop: 24,
            marginBottom: 56,
            maxWidth: 560,
            fontFamily: "var(--mono)",
            fontSize: 13,
            lineHeight: 1.6,
            letterSpacing: "0.02em",
            color: "var(--fg-dim)",
          }}
        >
          ChemGPT-4.7M (autoregressive Transformer trained on PubChem10M
          SELFIES) with a LoRA adapter biased toward self-healing polymer
          continuations. Submit a SMILES seed; candidates are scored on
          disulfide / urea / H-bond motifs and synthetic accessibility.
        </p>

        <HeroInput onSubmit={onSubmit} />
      </div>

      <Telemetry />
    </section>
  );
}

// ─── Generating overlay ───────────────────────────────────────────────────────

function GeneratingOverlay({
  seed,
  complete,
  error,
  onDone,
}: {
  seed: string;
  complete: boolean;
  error: string | null;
  onDone: () => void;
}) {
  const [phase, setPhase] = useState(0);
  const [progress, setProgress] = useState(0);
  const progressRef = useRef(0);
  const completeRef = useRef(complete);

  useEffect(() => {
    completeRef.current = complete;
  }, [complete]);

  useEffect(() => {
    if (error) return;
    const id = setInterval(() => {
      const isComplete = completeRef.current;
      const ceiling = isComplete ? 100 : 88;
      const speed = isComplete
        ? 6 + Math.random() * 5
        : 1.0 + Math.random() * 1.4;
      progressRef.current = Math.min(ceiling, progressRef.current + speed);
      const p = progressRef.current;
      if (p >= 100) {
        setProgress(100);
        clearInterval(id);
        setTimeout(onDone, 350);
        return;
      }
      setProgress(p);
      setPhase(Math.min(PHASES.length - 1, Math.floor(p / 20)));
    }, 90);
    return () => clearInterval(id);
  }, [error, onDone]);

  useEffect(() => {
    if (!error) return;
    const t = setTimeout(onDone, 2800);
    return () => clearTimeout(t);
  }, [error, onDone]);

  const errColor = "#ff5577";
  const accentColor = error ? errColor : ACCENT;

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(5,6,7,0.92)",
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
        zIndex: 100,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        animation: "fadeIn 0.25s ease",
      }}
    >
      <div style={{ width: "min(640px, 90vw)" }}>
        <div
          style={{
            fontFamily: "var(--mono)",
            fontSize: 10,
            letterSpacing: "0.2em",
            textTransform: "uppercase",
            color: "var(--fg-dim)",
            marginBottom: 24,
            display: "flex",
            justifyContent: "space-between",
          }}
        >
          <span>
            <BlinkDot />{" "}
            {error ? "GENERATION_FAILED" : "GENERATING_CANDIDATES"}
          </span>
          <span style={{ color: accentColor }}>
            {error
              ? "ERR"
              : `${Math.floor(progress).toString().padStart(3, "0")}%`}
          </span>
        </div>

        <div
          style={{
            fontFamily: "var(--mono)",
            fontSize: 13,
            color: "var(--fg)",
            marginBottom: 8,
            wordBreak: "break-all",
          }}
        >
          <span style={{ color: "var(--fg-dim)" }}>seed: </span>
          {seed || "(unconditional)"}
        </div>

        {/* Progress bar */}
        <div
          style={{
            height: 2,
            background: "var(--border)",
            marginTop: 32,
            marginBottom: 32,
            position: "relative",
          }}
        >
          <div
            style={{
              position: "absolute",
              left: 0,
              top: 0,
              bottom: 0,
              width: `${progress}%`,
              background: accentColor,
              boxShadow: `0 0 12px ${accentColor}`,
              transition: "width 0.15s linear",
            }}
          />
        </div>

        {error ? (
          <div
            style={{
              fontFamily: "var(--mono)",
              fontSize: 11,
              lineHeight: 1.7,
              color: "#ff8899",
              padding: 16,
              border: "1px solid #44222a",
              background: "rgba(60,10,20,0.3)",
              wordBreak: "break-word",
            }}
          >
            <div
              style={{
                color: errColor,
                marginBottom: 8,
                letterSpacing: "0.18em",
                textTransform: "uppercase",
                fontSize: 10,
              }}
            >
              ▸ ERROR
            </div>
            {error}
            <div
              style={{
                marginTop: 12,
                color: "var(--fg-dim)",
                fontSize: 10,
                letterSpacing: "0.16em",
                textTransform: "uppercase",
              }}
            >
              Is the backend running on {API_URL}?
            </div>
          </div>
        ) : (
          /* Phase log */
          <div
            style={{
              fontFamily: "var(--mono)",
              fontSize: 11,
              lineHeight: 1.9,
              color: "var(--fg-dim)",
              letterSpacing: "0.05em",
            }}
          >
            {PHASES.map((p, i) => {
              const done = i < phase;
              const active = i === phase;
              return (
                <div
                  key={p}
                  style={{
                    color: done
                      ? "var(--fg)"
                      : active
                        ? ACCENT
                        : "var(--fg-dimmer)",
                    opacity: i > phase ? 0.4 : 1,
                  }}
                >
                  <span style={{ display: "inline-block", width: 22 }}>
                    {done ? "✓" : active ? "▸" : "·"}
                  </span>
                  {p}
                  {active && <span className="blink"> _</span>}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Candidate card ───────────────────────────────────────────────────────────

function Stat({
  label,
  value,
  bar,
}: {
  label: string;
  value: string;
  bar?: number;
}) {
  return (
    <div
      style={{
        background: "rgba(8,11,13,0.8)",
        padding: "12px 14px",
        position: "relative",
      }}
    >
      <div
        style={{
          fontFamily: "var(--mono)",
          fontSize: 9,
          letterSpacing: "0.18em",
          color: "var(--fg-dim)",
          marginBottom: 4,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontFamily: "var(--mono)",
          fontSize: 16,
          color: "var(--fg)",
          letterSpacing: "0.01em",
        }}
      >
        {value}
      </div>
      {bar !== undefined && (
        <div
          style={{
            position: "absolute",
            left: 0,
            bottom: 0,
            height: 2,
            width: `${bar}%`,
            background: ACCENT,
          }}
        />
      )}
    </div>
  );
}

function CandidateCard({
  c,
  index,
  total,
}: {
  c: Candidate;
  index: number;
  total: number;
}) {
  const [expanded, setExpanded] = useState(false);
  const [explanation, setExplanation] = useState<string | null>(null);
  const [explanationError, setExplanationError] = useState<string | null>(null);

  const motifList =
    c.motifs_found && c.motifs_found.length > 0
      ? c.motifs_found.map((m) => m.replace(/_/g, " ")).join(", ")
      : "none detected";

  const flags: string[] = [];
  if (c.has_strict_sh) flags.push("STRICT-SH");
  if (c.is_polymer) flags.push("POLYMER");
  if (c.synthesisable === true) flags.push("SYNTHESISABLE");
  else if (c.synthesisable === false) flags.push("LOW SYNTH");
  const flagText = flags.length ? flags.join(" · ") : "—";

  const synthLabel =
    c.sa_score == null
      ? "N/A"
      : c.sa_score <= 4
        ? "easy"
        : c.sa_score <= 6
          ? "moderate"
          : "hard";

  const borderColor = c.valid ? "var(--border)" : "#44222a";
  const bracketColor = c.valid ? "var(--border-strong)" : "#66333d";
  const idColor = c.valid ? ACCENT : "#ff8899";

  // Valid → show canonical SMILES; invalid → show raw SELFIES so user can
  // see what the model actually emitted.
  const displaySmiles = c.valid
    ? c.canonical_smiles || c.smiles || "(empty output)"
    : c.selfies || "(empty output)";

  return (
    <div
      style={{
        border: `1px solid ${borderColor}`,
        background: "rgba(10,13,16,0.6)",
        padding: 24,
        position: "relative",
        animation: `slideUp 0.5s ${index * 0.08}s both ease-out`,
      }}
    >
      <Bracket pos="tl" color={bracketColor} />
      <Bracket pos="tr" color={bracketColor} />
      <Bracket pos="bl" color={bracketColor} />
      <Bracket pos="br" color={bracketColor} />

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 14,
        }}
      >
        <div
          style={{
            fontFamily: "var(--mono)",
            fontSize: 11,
            letterSpacing: "0.18em",
            color: idColor,
          }}
        >
          {c.id}
          {!c.valid && (
            <span style={{ marginLeft: 12, color: "#ff8899" }}>
              · INVALID SMILES
            </span>
          )}
        </div>
        <div
          style={{
            fontFamily: "var(--mono)",
            fontSize: 10,
            letterSpacing: "0.12em",
            color: "var(--fg-dim)",
          }}
        >
          #{index + 1}
          <span style={{ opacity: 0.5 }}> / {total}</span>
        </div>
      </div>

      {c.valid && (
        <div style={{ marginBottom: 16 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontFamily: "var(--mono)",
              fontSize: 9,
              letterSpacing: "0.18em",
              color: "var(--fg-dim)",
              marginBottom: 6,
            }}
          >
            <span>SELF-HEALING QUALITY</span>
            <span style={{ color: ACCENT }}>{c.shScore}/100</span>
          </div>
          <div
            style={{
              height: 4,
              background: "var(--border)",
              position: "relative",
            }}
          >
            <div
              style={{
                position: "absolute",
                left: 0,
                top: 0,
                bottom: 0,
                width: `${c.shScore}%`,
                background: ACCENT,
                boxShadow: c.shScore > 0 ? `0 0 8px ${ACCENT}88` : undefined,
                transition: "width 0.3s",
              }}
            />
          </div>
        </div>
      )}

      <div
        style={{
          fontFamily: "var(--mono)",
          fontSize: 11,
          lineHeight: 1.55,
          color: "var(--fg)",
          wordBreak: "break-all",
          padding: 12,
          background: "rgba(0,0,0,0.4)",
          border: "1px solid var(--border)",
          marginBottom: 16,
        }}
      >
        {displaySmiles}
      </div>

      {c.valid ? (
        <>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(3, 1fr)",
              gap: 1,
              background: "var(--border)",
              border: "1px solid var(--border)",
              marginBottom: 16,
            }}
          >
            <Stat
              label="MW (g/mol)"
              value={
                c.molecular_weight != null
                  ? c.molecular_weight.toFixed(1)
                  : "—"
              }
            />
            <Stat
              label={`SYNTH · ${synthLabel}`}
              value={c.sa_score != null ? c.sa_score.toFixed(2) : "N/A"}
            />
            <Stat
              label="RINGS · ATOMS"
              value={`${c.num_rings ?? "—"} · ${c.num_heavy_atoms ?? "—"}`}
            />
          </div>

          <div
            style={{
              fontFamily: "var(--mono)",
              fontSize: 10,
              letterSpacing: "0.16em",
              textTransform: "uppercase",
              color: "var(--fg-dim)",
              lineHeight: 1.9,
            }}
          >
            <div>
              MOTIFS ·{" "}
              <span
                style={{
                  color: c.has_any_sh_motif ? ACCENT : "var(--fg)",
                }}
              >
                {motifList}
              </span>
            </div>
            <div>
              FLAGS · <span style={{ color: "var(--fg)" }}>{flagText}</span>
            </div>
          </div>
        </>
      ) : (
        <div
          style={{
            fontFamily: "var(--mono)",
            fontSize: 11,
            color: "#ff8899",
            padding: 12,
            border: "1px solid #44222a",
            background: "rgba(60,10,20,0.2)",
          }}
        >
          ▸ RDKit could not parse this output as a valid molecule.
        </div>
      )}

      <button
        type="button"
        onClick={() => setExpanded(true)}
        style={{
          width: "100%",
          marginTop: 16,
          padding: "8px 0",
          background: "transparent",
          border: "none",
          borderTop: "1px solid var(--border)",
          color: "var(--fg-dim)",
          fontFamily: "var(--mono)",
          fontSize: 10,
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          cursor: "pointer",
          transition: "color 0.2s",
        }}
        onMouseEnter={(e) => (e.currentTarget.style.color = ACCENT)}
        onMouseLeave={(e) => (e.currentTarget.style.color = "var(--fg-dim)")}
      >
        VIEW DETAILS →
      </button>

      {expanded && (
        <CandidateModal
          c={c}
          onClose={() => setExpanded(false)}
          cachedExplanation={explanation}
          cachedExplanationError={explanationError}
          onExplanation={setExplanation}
          onExplanationError={setExplanationError}
        />
      )}
    </div>
  );
}

function CandidateModal({
  c,
  onClose,
  cachedExplanation,
  cachedExplanationError,
  onExplanation,
  onExplanationError,
}: {
  c: Candidate;
  onClose: () => void;
  cachedExplanation: string | null;
  cachedExplanationError: string | null;
  onExplanation: (text: string) => void;
  onExplanationError: (msg: string) => void;
}) {
  // ESC to close + lock page scroll while open. Lock both <html> and <body>
  // because browsers differ on which one owns the document scrollbar.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    const html = document.documentElement;
    const body = document.body;
    const prevHtml = html.style.overflow;
    const prevBody = body.style.overflow;
    html.style.overflow = "hidden";
    body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      html.style.overflow = prevHtml;
      body.style.overflow = prevBody;
    };
  }, [onClose]);

  const breakdown = c.valid ? selfHealingBreakdown(c) : [];
  const synthLabel =
    c.sa_score == null
      ? "N/A"
      : c.sa_score <= 4
        ? "easy"
        : c.sa_score <= 6
          ? "moderate"
          : "hard";

  // Fetch AI explanation when modal opens — only if not already cached
  // (cache lives on the parent CandidateCard so it survives modal remount).
  // Skip when candidate is invalid (no SMILES worth explaining).
  const hasSmilesToExplain = c.valid && Boolean(c.canonical_smiles ?? c.smiles);
  const explanationLoading =
    hasSmilesToExplain &&
    cachedExplanation === null &&
    cachedExplanationError === null;

  useEffect(() => {
    if (!explanationLoading) return;
    let cancelled = false;
    explainMolecule(c)
      .then((text) => {
        if (!cancelled) onExplanation(text);
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          onExplanationError(e instanceof Error ? e.message : String(e));
        }
      });
    return () => {
      cancelled = true;
    };
    // We intentionally only kick this off once per mount — cache lives in the
    // parent, so a remount with cache populated will not re-enter the branch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // SSR guard — document only exists in the browser
  if (typeof document === "undefined") return null;

  return createPortal(
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(5,6,7,0.88)",
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
        zIndex: 50,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "5vh 5vw",
        animation: "fadeIn 0.2s ease",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="no-scrollbar"
        style={{
          position: "relative",
          background: "rgba(10,13,16,0.97)",
          border: "1px solid var(--border-strong)",
          width: "100%",
          maxWidth: 1100,
          height: "90vh",
          overflowY: "auto",
          overflowX: "hidden",
          padding: 40,
          animation: "slideUp 0.3s ease",
        }}
      >
        <Bracket pos="tl" color={ACCENT} />
        <Bracket pos="tr" color={ACCENT} />
        <Bracket pos="bl" color={ACCENT} />
        <Bracket pos="br" color={ACCENT} />

        {/* Header */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "baseline",
            marginBottom: 20,
            paddingBottom: 16,
            borderBottom: "1px solid var(--border)",
          }}
        >
          <div
            style={{
              fontFamily: "var(--mono)",
              fontSize: 11,
              letterSpacing: "0.18em",
              color: c.valid ? ACCENT : "#ff8899",
            }}
          >
            {c.id} · CANDIDATE DETAILS
            {!c.valid && (
              <span style={{ marginLeft: 12, color: "#ff8899" }}>
                · INVALID
              </span>
            )}
          </div>
          <button
            type="button"
            onClick={onClose}
            style={{
              background: "transparent",
              border: "1px solid var(--border)",
              color: "var(--fg-dim)",
              padding: "6px 12px",
              fontFamily: "var(--mono)",
              fontSize: 10,
              letterSpacing: "0.18em",
              textTransform: "uppercase",
              cursor: "pointer",
              transition: "color 0.2s, border-color 0.2s",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = ACCENT;
              e.currentTarget.style.borderColor = ACCENT;
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = "var(--fg-dim)";
              e.currentTarget.style.borderColor = "var(--border)";
            }}
          >
            ✕ Close · ESC
          </button>
        </div>

        {/* SH Quality bar — repeated in the modal so it's always visible */}
        {c.valid && (
          <div style={{ marginBottom: 8 }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontFamily: "var(--mono)",
                fontSize: 9,
                letterSpacing: "0.18em",
                color: "var(--fg-dim)",
                marginBottom: 6,
              }}
            >
              <span>SELF-HEALING QUALITY</span>
              <span style={{ color: ACCENT }}>{c.shScore}/100</span>
            </div>
            <div
              style={{
                height: 4,
                background: "var(--border)",
                position: "relative",
              }}
            >
              <div
                style={{
                  position: "absolute",
                  left: 0,
                  top: 0,
                  bottom: 0,
                  width: `${c.shScore}%`,
                  background: ACCENT,
                  boxShadow: c.shScore > 0 ? `0 0 8px ${ACCENT}88` : undefined,
                }}
              />
            </div>
          </div>
        )}

        <SectionHeader>Representations</SectionHeader>
        <DefList
          rows={[
            ["CANONICAL", c.canonical_smiles ?? "—"],
            ["SMILES", c.smiles ?? "—"],
            ["SELFIES", c.selfies || "—"],
          ]}
          mono
        />

        {c.valid ? (
          <>
            <SectionHeader>SH Quality breakdown</SectionHeader>
            <div
              style={{
                fontFamily: "var(--mono)",
                fontSize: 11,
                lineHeight: 1.8,
              }}
            >
              {breakdown.map((b) => (
                <div
                  key={b.label}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "180px 80px 1fr",
                    gap: 12,
                    color: b.awarded > 0 ? "var(--fg)" : "var(--fg-dimmer)",
                  }}
                >
                  <span>{b.label}</span>
                  <span
                    style={{
                      color: b.awarded > 0 ? ACCENT : "var(--fg-dimmer)",
                    }}
                  >
                    +{b.awarded} / {b.max}
                  </span>
                  <span style={{ color: "var(--fg-dim)" }}>{b.reason}</span>
                </div>
              ))}
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "180px 80px 1fr",
                  gap: 12,
                  marginTop: 8,
                  paddingTop: 8,
                  borderTop: "1px solid var(--border)",
                  color: "var(--fg)",
                }}
              >
                <span>total</span>
                <span style={{ color: ACCENT }}>{c.shScore} / 100</span>
                <span />
              </div>
            </div>

            <SectionHeader>Descriptors</SectionHeader>
            <DefList
              rows={[
                [
                  "Molecular weight",
                  c.molecular_weight != null
                    ? `${c.molecular_weight.toFixed(2)} g/mol`
                    : "—",
                ],
                ["Heavy atoms", String(c.num_heavy_atoms ?? "—")],
                ["Rings", String(c.num_rings ?? "—")],
                [
                  "SA score",
                  c.sa_score != null
                    ? `${c.sa_score.toFixed(2)} (${synthLabel})`
                    : "N/A",
                ],
                ["Polymer flag", c.is_polymer ? "yes" : "no"],
                [
                  "Synthesisable",
                  c.synthesisable === true
                    ? "yes"
                    : c.synthesisable === false
                      ? "no"
                      : "N/A",
                ],
                [
                  "Motifs detected",
                  c.motifs_found && c.motifs_found.length > 0
                    ? c.motifs_found.join(", ")
                    : "none",
                ],
              ]}
            />
          </>
        ) : (
          <div
            style={{
              marginTop: 16,
              fontFamily: "var(--mono)",
              fontSize: 11,
              color: "var(--fg-dim)",
              padding: 12,
              border: "1px solid #44222a",
              background: "rgba(60,10,20,0.2)",
            }}
          >
            ▸ Metrics unavailable — RDKit could not parse the decoded SMILES.
          </div>
        )}

        <SectionHeader>Structure</SectionHeader>
        <MoleculeStructure smiles={c.canonical_smiles ?? c.smiles ?? null} />

        <AIExplanationPanel
          loading={explanationLoading}
          error={cachedExplanationError}
          text={cachedExplanation}
          disabled={!hasSmilesToExplain}
        />
      </div>
    </div>,
    document.body,
  );
}

const MOL_W = 520;
const MOL_H = 280;
const ZOOM_MIN = 0.5;
const ZOOM_MAX = 5;
const ZOOM_STEP = 0.25;

function MoleculeStructure({ smiles }: { smiles: string | null }) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [error, setError] = useState(false);
  const [zoom, setZoom] = useState(1);

  // Reset zoom whenever a new SMILES is rendered (opening a different modal
  // shouldn't inherit the previous candidate's zoom level).
  useEffect(() => {
    setZoom(1);
  }, [smiles]);

  useEffect(() => {
    if (!smiles || smiles === "INVALID" || !svgRef.current) return;
    setError(false);
    let cancelled = false;
    (async () => {
      // smiles-drawer's src/ tree is unbuildable (mixes .js with .ts files);
      // load the pre-built IIFE dist bundle instead, which registers
      // `window.SmilesDrawer`. The dist file has no types of its own.
      // @ts-expect-error - untyped browser bundle, declares only via side effect
      await import("smiles-drawer/dist/smiles-drawer.min.js");
      if (cancelled) return;
      const SD = window.SmilesDrawer;
      if (!SD || !svgRef.current) {
        setError(true);
        return;
      }
      const drawer = new SD.SvgDrawer({
        width: MOL_W,
        height: MOL_H,
        bondThickness: 1.1,
        padding: 12,
      });
      SD.parse(
        smiles,
        (tree) => {
          if (cancelled || !svgRef.current) return;
          svgRef.current.innerHTML = "";
          drawer.draw(tree, svgRef.current, "dark");
        },
        () => {
          if (!cancelled) setError(true);
        },
      );
    })();
    return () => {
      cancelled = true;
    };
  }, [smiles]);

  if (!smiles || smiles === "INVALID") {
    return (
      <div
        style={{
          fontFamily: "var(--mono)",
          fontSize: 11,
          color: "var(--fg-dim)",
          padding: 12,
          border: "1px dashed var(--border)",
        }}
      >
        ▸ No structure preview — SMILES could not be decoded.
      </div>
    );
  }

  const clamp = (z: number) =>
    Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, Math.round(z * 100) / 100));

  const btnStyle: React.CSSProperties = {
    fontFamily: "var(--mono)",
    fontSize: 12,
    lineHeight: "20px",
    width: 24,
    height: 24,
    background: "rgba(0,0,0,0.55)",
    color: "var(--fg)",
    border: "1px solid var(--border)",
    cursor: "pointer",
    padding: 0,
  };

  return (
    <div
      style={{
        position: "relative",
        padding: 16,
        background: "rgba(0,0,0,0.25)",
        border: "1px solid var(--border)",
        minHeight: 280,
      }}
    >
      {!error && (
        <div
          style={{
            position: "absolute",
            top: 8,
            right: 8,
            display: "flex",
            alignItems: "center",
            gap: 4,
            zIndex: 1,
            fontFamily: "var(--mono)",
            fontSize: 11,
            color: "var(--fg-dim)",
          }}
        >
          <button
            type="button"
            aria-label="Zoom out"
            onClick={() => setZoom((z) => clamp(z - ZOOM_STEP))}
            disabled={zoom <= ZOOM_MIN}
            style={{
              ...btnStyle,
              opacity: zoom <= ZOOM_MIN ? 0.35 : 1,
              cursor: zoom <= ZOOM_MIN ? "default" : "pointer",
            }}
          >
            −
          </button>
          <span
            style={{
              minWidth: 38,
              textAlign: "center",
              userSelect: "none",
            }}
          >
            {Math.round(zoom * 100)}%
          </span>
          <button
            type="button"
            aria-label="Reset zoom"
            onClick={() => setZoom(1)}
            disabled={zoom === 1}
            style={{
              ...btnStyle,
              opacity: zoom === 1 ? 0.35 : 1,
              cursor: zoom === 1 ? "default" : "pointer",
            }}
          >
            ⌂
          </button>
          <button
            type="button"
            aria-label="Zoom in"
            onClick={() => setZoom((z) => clamp(z + ZOOM_STEP))}
            disabled={zoom >= ZOOM_MAX}
            style={{
              ...btnStyle,
              opacity: zoom >= ZOOM_MAX ? 0.35 : 1,
              cursor: zoom >= ZOOM_MAX ? "default" : "pointer",
            }}
          >
            +
          </button>
        </div>
      )}

      {error ? (
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            minHeight: 248,
            fontFamily: "var(--mono)",
            fontSize: 11,
            color: "var(--fg-dim)",
          }}
        >
          ▸ Could not render structure for: {smiles}
        </div>
      ) : (
        // Scroll viewport. `margin: auto` on the inner sizing wrapper centers
        // it when smaller than the viewport, but yields the top-left when
        // larger — so zoomed-in content stays scrollable instead of being
        // clipped off-screen (classic flex-centering-with-overflow fix).
        <div
          style={{
            width: "100%",
            height: 280,
            overflow: "auto",
            display: "flex",
          }}
        >
          <div
            style={{
              width: MOL_W * zoom,
              height: MOL_H * zoom,
              flexShrink: 0,
              margin: "auto",
            }}
          >
            <svg
              ref={svgRef}
              width={MOL_W}
              height={MOL_H}
              role="img"
              aria-label={`2D structure of ${smiles}`}
              style={{
                transform: `scale(${zoom})`,
                transformOrigin: "top left",
                display: "block",
                transition: "transform 0.12s ease-out",
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function AIExplanationPanel({
  loading,
  error,
  text,
  disabled,
}: {
  loading: boolean;
  error: string | null;
  text: string | null;
  disabled: boolean;
}) {
  return (
    <div
      style={{
        marginTop: 20,
        padding: 18,
        border: `1px solid ${ACCENT}40`,
        background: "rgba(0,255,157,0.04)",
        position: "relative",
      }}
    >
      <div
        style={{
          fontFamily: "var(--mono)",
          fontSize: 10,
          letterSpacing: "0.24em",
          textTransform: "uppercase",
          color: ACCENT,
          marginBottom: 12,
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <span style={{ color: ACCENT }}>▌</span>
        AI ANALYSIS
        <span style={{ color: "var(--fg-dimmer)", fontSize: 9 }}>
          · claude-sonnet-4
        </span>
        {loading && (
          <span
            className="blink"
            style={{ color: ACCENT, marginLeft: 4, fontSize: 9 }}
          >
            ●
          </span>
        )}
      </div>

      {disabled ? (
        <div
          style={{
            fontFamily: "var(--mono)",
            fontSize: 11,
            color: "var(--fg-dim)",
          }}
        >
          ▸ Skipped — no decoded SMILES to analyse.
        </div>
      ) : loading ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {[100, 92, 78].map((w, i) => (
            <div
              key={i}
              style={{
                height: 12,
                width: `${w}%`,
                background: `linear-gradient(90deg, ${ACCENT}22, ${ACCENT}11)`,
                animation: "pulse 1.4s ease-in-out infinite",
                animationDelay: `${i * 0.15}s`,
              }}
            />
          ))}
        </div>
      ) : error ? (
        <div
          style={{
            fontFamily: "var(--mono)",
            fontSize: 11,
            color: "#ff8899",
          }}
        >
          ▸ Could not load explanation: {error}
        </div>
      ) : text ? (
        <div
          style={{
            fontFamily: "var(--mono)",
            fontSize: 12.5,
            lineHeight: 1.7,
            color: "var(--fg)",
            whiteSpace: "pre-wrap",
          }}
        >
          {text}
        </div>
      ) : null}
    </div>
  );
}

function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        fontFamily: "var(--mono)",
        fontSize: 9,
        letterSpacing: "0.22em",
        textTransform: "uppercase",
        color: ACCENT,
        marginTop: 16,
        marginBottom: 10,
        paddingBottom: 6,
        borderBottom: "1px dashed var(--border)",
      }}
    >
      ── {children} ──
    </div>
  );
}

function DefList({
  rows,
  mono,
}: {
  rows: [string, string][];
  mono?: boolean;
}) {
  return (
    <div
      style={{
        fontFamily: "var(--mono)",
        fontSize: 11,
        lineHeight: 1.8,
      }}
    >
      {rows.map(([label, value]) => (
        <div
          key={label}
          style={{
            display: "grid",
            gridTemplateColumns: "180px 1fr",
            gap: 12,
          }}
        >
          <span style={{ color: "var(--fg-dim)" }}>{label}</span>
          <span
            style={{
              color: "var(--fg)",
              wordBreak: mono ? "break-all" : "normal",
            }}
          >
            {value}
          </span>
        </div>
      ))}
    </div>
  );
}

// ─── Results ──────────────────────────────────────────────────────────────────

function BatchStrip({ b }: { b: BatchMetrics | null }) {
  if (!b) return null;
  // Note: diversity / mean_sa_score / synthesisable_% can be the literal
  // string "N/A" (when SA scorer is unavailable or <2 valid molecules) — the
  // formatMetric helper handles both number and string cases.
  const items: { k: string; v: string; accent?: boolean }[] = [
    {
      k: "VALIDITY",
      v: `${b["validity_%"]}% (${b.n_valid}/${b.n_total})`,
    },
    { k: "UNIQUE", v: `${b["uniqueness_%"]}%` },
    {
      k: "SH MOTIF",
      v: `${b["self_healing_motif_%"]}%`,
      accent: b["self_healing_motif_%"] > 0,
    },
    {
      k: "STRICT SH",
      v: `${b["strict_sh_%"]}%`,
      accent: b["strict_sh_%"] > 0,
    },
    { k: "MEAN SA", v: formatMetric(b.mean_sa_score, 2) },
    { k: "DIVERSITY", v: formatMetric(b.diversity, 3) },
  ];
  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        marginBottom: 32,
        border: "1px solid var(--border)",
        background: "rgba(10,13,16,0.6)",
      }}
    >
      {items.map((it, i) => (
        <div
          key={it.k}
          style={{
            flex: "1 1 120px",
            padding: "14px 18px",
            borderRight:
              i < items.length - 1 ? "1px solid var(--border)" : "none",
          }}
        >
          <div
            style={{
              fontFamily: "var(--mono)",
              fontSize: 9,
              letterSpacing: "0.18em",
              color: "var(--fg-dim)",
              marginBottom: 4,
            }}
          >
            {it.k}
          </div>
          <div
            style={{
              fontFamily: "var(--mono)",
              fontSize: 14,
              color: it.accent ? ACCENT : "var(--fg)",
            }}
          >
            {it.v}
          </div>
        </div>
      ))}
    </div>
  );
}

function Results({
  seed,
  candidates,
  batchMetrics,
  onClose,
}: {
  seed: string;
  candidates: Candidate[];
  batchMetrics: BatchMetrics | null;
  onClose: () => void;
}) {
  const batch = candidates[0]?.id.replace("MTZ-", "").slice(0, 3) ?? "000";
  const count = String(candidates.length).padStart(2, "0");

  return (
    <section
      style={{
        position: "relative",
        zIndex: 2,
        padding: "80px 40px 120px",
        borderTop: "1px solid var(--border)",
        background: "linear-gradient(to bottom, transparent, rgba(5,6,7,0.6))",
        animation: "fadeIn 0.5s ease",
      }}
    >
      <div style={{ maxWidth: 1180, margin: "0 auto" }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "baseline",
            marginBottom: 32,
            gap: 24,
          }}
        >
          <div style={{ minWidth: 0, flex: 1 }}>
            <div
              style={{
                fontFamily: "var(--mono)",
                fontSize: 10,
                letterSpacing: "0.2em",
                textTransform: "uppercase",
                color: "var(--fg-dim)",
                marginBottom: 8,
              }}
            >
              <BlinkDot /> {count} CANDIDATES · BATCH MTZ-{batch}
            </div>
            <h2
              style={{
                fontFamily: "var(--display)",
                fontSize: 32,
                fontWeight: 400,
                letterSpacing: "-0.02em",
                margin: 0,
                color: "var(--fg)",
              }}
            >
              Top-ranked novel polymers
            </h2>
            <div
              style={{
                fontFamily: "var(--mono)",
                fontSize: 11,
                color: "var(--fg-dim)",
                marginTop: 10,
                wordBreak: "break-all",
              }}
            >
              <span style={{ color: ACCENT }}>seed → </span>
              {seed || "(unconditional)"}
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: "transparent",
              border: "1px solid var(--border)",
              color: "var(--fg-dim)",
              padding: "10px 16px",
              fontFamily: "var(--mono)",
              fontSize: 10,
              letterSpacing: "0.18em",
              textTransform: "uppercase",
              cursor: "pointer",
              flexShrink: 0,
              alignSelf: "flex-start",
            }}
          >
            ✕ Clear
          </button>
        </div>

        <BatchStrip b={batchMetrics} />

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))",
            gap: 24,
          }}
        >
          {candidates.map((c, i) => (
            <CandidateCard
              key={`${c.id}-${i}`}
              c={c}
              index={i}
              total={candidates.length}
            />
          ))}
        </div>
      </div>
    </section>
  );
}

// ─── How It Works ─────────────────────────────────────────────────────────────

function StepDiagram({ step }: { step: number }) {
  if (step === 0) {
    return (
      <div
        style={{
          height: 80,
          fontFamily: "var(--mono)",
          fontSize: 11,
          color: "var(--fg-dim)",
          display: "flex",
          alignItems: "center",
          gap: 10,
        }}
      >
        <span style={{ color: ACCENT }}>›</span>
        <span style={{ color: "var(--fg)" }}>OCCNC(=O)</span>
        <span>NCCSSCC</span>
        <span className="blink" style={{ color: ACCENT }}>
          _
        </span>
      </div>
    );
  }
  if (step === 1) {
    return (
      <svg viewBox="0 0 200 80" style={{ height: 80, width: "100%" }}>
        {Array.from({ length: 24 }).map((_, i) => {
          const x = (i % 8) * 24 + 8;
          const y = Math.floor(i / 8) * 24 + 8;
          const a = 0.15 + ((i * 37) % 100) / 100 * 0.7;
          return (
            <rect
              key={i}
              x={x}
              y={y}
              width="14"
              height="14"
              fill={ACCENT}
              opacity={a}
            />
          );
        })}
      </svg>
    );
  }
  const bars = [0.94, 0.88, 0.84, 0.71, 0.62];
  return (
    <div
      style={{
        height: 80,
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        gap: 6,
      }}
    >
      {bars.map((b, i) => (
        <div
          key={i}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            fontFamily: "var(--mono)",
            fontSize: 9,
            color: "var(--fg-dim)",
          }}
        >
          <span style={{ width: 28 }}>0{i + 1}</span>
          <div
            style={{
              flex: 1,
              height: 6,
              background: "var(--border)",
              position: "relative",
            }}
          >
            <div
              style={{
                position: "absolute",
                left: 0,
                top: 0,
                bottom: 0,
                width: `${b * 100}%`,
                background: i === 0 ? ACCENT : `${ACCENT}55`,
              }}
            />
          </div>
          <span
            style={{ color: i === 0 ? ACCENT : "var(--fg-dim)", width: 32 }}
          >
            {b.toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  );
}

function HowItWorks() {
  const steps = [
    {
      n: "01",
      title: "Seed",
      tag: "INPUT",
      body: "Submit a SMILES string — a known polymer, a fragment, or a designed motif. The seed is encoded as SELFIES tokens before generation.",
    },
    {
      n: "02",
      title: "Sample",
      tag: "INFERENCE",
      body: "An autoregressive Transformer (ChemGPT-4.7M) samples SELFIES tokens one at a time. The LoRA adapter shifts the next-token distribution toward self-healing polymer continuations.",
    },
    {
      n: "03",
      title: "Score",
      tag: "OUTPUT",
      body: "Outputs decode back to SMILES; RDKit validates them and seven SMARTS patterns check for self-healing motifs (disulfide, urea, amide H-bond, furan, imine, boronic ester, DA adduct). A composite 0–100 Self-Healing Quality score combines motif hits, polymer continuation, synthetic accessibility, and molecular weight.",
    },
  ];

  return (
    <section
      id="how"
      style={{
        position: "relative",
        padding: "160px 40px",
        borderTop: "1px solid var(--border)",
        background: "#070809",
      }}
    >
      <div style={{ maxWidth: 1180, margin: "0 auto" }}>
        <div
          style={{
            fontFamily: "var(--mono)",
            fontSize: 10,
            letterSpacing: "0.22em",
            textTransform: "uppercase",
            color: ACCENT,
            marginBottom: 16,
          }}
        >
          02 / METHOD
        </div>
        <h2
          style={{
            fontFamily: "var(--display)",
            fontSize: "clamp(36px, 4vw, 56px)",
            fontWeight: 400,
            letterSpacing: "-0.025em",
            lineHeight: 1.05,
            margin: 0,
            marginBottom: 80,
            maxWidth: 720,
            color: "var(--fg)",
          }}
        >
          From SMILES to synthesizable candidate in{" "}
          <span
            style={{
              color: ACCENT,
              fontStyle: "italic",
              fontWeight: 300,
            }}
          >
            seconds
          </span>
          .
        </h2>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            border: "1px solid var(--border)",
          }}
        >
          {steps.map((s, i) => (
            <div
              key={s.n}
              style={{
                padding: 40,
                borderRight:
                  i < 2 ? "1px solid var(--border)" : "none",
                position: "relative",
                minHeight: 320,
                display: "flex",
                flexDirection: "column",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontFamily: "var(--mono)",
                  fontSize: 10,
                  letterSpacing: "0.18em",
                  color: "var(--fg-dim)",
                  marginBottom: 32,
                }}
              >
                <span>{s.n}</span>
                <span style={{ color: ACCENT }}>{s.tag}</span>
              </div>

              <StepDiagram step={i} />

              <h3
                style={{
                  fontFamily: "var(--display)",
                  fontSize: 28,
                  fontWeight: 400,
                  letterSpacing: "-0.02em",
                  margin: "40px 0 12px",
                  color: "var(--fg)",
                }}
              >
                {s.title}
              </h3>
              <p
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: 12,
                  lineHeight: 1.7,
                  color: "var(--fg-dim)",
                  margin: 0,
                }}
              >
                {s.body}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ─── Footer ───────────────────────────────────────────────────────────────────

function Footer() {
  const papers = [
    {
      id: "ncfrey",
      title: "ChemGPT-4.7M",
      authors: "ncfrey",
      venue: "HuggingFace",
      href: "https://huggingface.co/ncfrey/ChemGPT-4.7M",
    },
    {
      id: "2020.10",
      title: "Self-Referencing Embedded Strings (SELFIES): A 100% robust molecular string representation",
      authors: "Krenn et al.",
      venue: "Mach. Learn.: Sci. Technol.",
      href: "https://iopscience.iop.org/article/10.1088/2632-2153/aba947",
    },
    {
      id: "2020.09",
      title: "PI1M: A Benchmark Database for Polymer Informatics",
      authors: "Ma, Luo",
      venue: "ACS Publications",
      href: "https://pubs.acs.org/doi/10.1021/acs.jcim.0c00726",
    },
    {
      id: "lora",
      title: "LoRA: Low-Rank Adaptation of Large Language Models",
      authors: "Hu et al.",
      venue: "arXiv 2106.09685",
      href: "https://arxiv.org/abs/2106.09685",
    },
  ];

  return (
    <footer
      id="research"
      style={{
        position: "relative",
        padding: "160px 40px 60px",
        borderTop: "1px solid var(--border)",
        background: "#050607",
      }}
    >
      <div style={{ maxWidth: 1180, margin: "0 auto" }}>
        <div
          style={{
            fontFamily: "var(--mono)",
            fontSize: 10,
            letterSpacing: "0.22em",
            textTransform: "uppercase",
            color: ACCENT,
            marginBottom: 16,
          }}
        >
          03 / RESEARCH
        </div>
        <h2
          style={{
            fontFamily: "var(--display)",
            fontSize: "clamp(32px, 3.5vw, 48px)",
            fontWeight: 400,
            letterSpacing: "-0.025em",
            margin: 0,
            marginBottom: 56,
            maxWidth: 640,
            color: "var(--fg)",
          }}
        >
          Open work from the lab.
        </h2>

        <div style={{ borderTop: "1px solid var(--border)" }}>
          {papers.map((p) => (
            <a
              key={p.id}
              href={p.href}
              target={p.href !== "#" ? "_blank" : undefined}
              rel={p.href !== "#" ? "noopener noreferrer" : undefined}
              style={{
                display: "grid",
                gridTemplateColumns: "120px 1fr 200px 80px",
                gap: 24,
                padding: "24px 0",
                borderBottom: "1px solid var(--border)",
                textDecoration: "none",
                color: "inherit",
                alignItems: "baseline",
                transition: "background 0.2s",
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.background = "rgba(255,255,255,0.02)")
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.background = "transparent")
              }
            >
              <span
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: 11,
                  letterSpacing: "0.16em",
                  color: ACCENT,
                }}
              >
                {p.id}
              </span>
              <span
                style={{
                  fontFamily: "var(--display)",
                  fontSize: 18,
                  color: "var(--fg)",
                  letterSpacing: "-0.01em",
                }}
              >
                {p.title}
              </span>
              <span
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: 11,
                  color: "var(--fg-dim)",
                }}
              >
                {p.authors}
              </span>
              <span
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: 10,
                  letterSpacing: "0.16em",
                  textTransform: "uppercase",
                  color: "var(--fg-dim)",
                  textAlign: "right",
                }}
              >
                {p.venue} →
              </span>
            </a>
          ))}
        </div>

        <div
          style={{
            marginTop: 80,
            paddingTop: 32,
            borderTop: "1px solid var(--border)",
            display: "flex",
            justifyContent: "space-between",
            fontFamily: "var(--mono)",
            fontSize: 10,
            letterSpacing: "0.16em",
            textTransform: "uppercase",
            color: "var(--fg-dim)",
            alignItems: "center",
          }}
        >
          <Logo />
          <span>© 2026 · Materialaize Lab · Monash</span>
          <span>
            STATUS <span style={{ color: ACCENT }}>● OPERATIONAL</span>
          </span>
        </div>
      </div>
    </footer>
  );
}

// ─── Root ─────────────────────────────────────────────────────────────────────

type ResultState = {
  seed: string;
  pending: boolean;
  candidates: Candidate[];
  batchMetrics: BatchMetrics | null;
  error: string | null;
} | null;

export default function Landing() {
  const [generating, setGenerating] = useState(false);
  const [results, setResults] = useState<ResultState>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const requestIdRef = useRef(0);

  const onSubmit = useCallback((seed: string) => {
    const myRequestId = ++requestIdRef.current;
    setGenerating(true);
    setResults({
      seed,
      pending: true,
      candidates: [],
      batchMetrics: null,
      error: null,
    });

    generateMolecules(seed, 3)
      .then(({ molecules, batchMetrics }) => {
        if (myRequestId !== requestIdRef.current) return;
        if (molecules.length === 0) {
          setResults({
            seed,
            pending: true,
            candidates: [],
            batchMetrics,
            error: "Backend returned no molecules.",
          });
          return;
        }
        const list = molecules
          .map(toCandidate)
          .sort((a, b) => b.shScore - a.shScore);
        setResults({
          seed,
          pending: true,
          candidates: list,
          batchMetrics,
          error: null,
        });
      })
      .catch((e: unknown) => {
        if (myRequestId !== requestIdRef.current) return;
        const message = e instanceof Error ? e.message : String(e);
        setResults({
          seed,
          pending: true,
          candidates: [],
          batchMetrics: null,
          error: message,
        });
      });
  }, []);

  const onGenDone = useCallback(() => {
    setGenerating(false);
    setResults((prev) => {
      if (!prev) return null;
      if (prev.error) return null;
      return { ...prev, pending: false };
    });
    setTimeout(() => {
      if (resultsRef.current) {
        window.scrollTo({
          top: resultsRef.current.offsetTop - 40,
          behavior: "smooth",
        });
      }
    }, 100);
  }, []);

  return (
    <div style={{ minHeight: "100vh", background: "#050607", color: "#e8eae6" }}>
      {/* Fixed animated background */}
      <div
        style={{ position: "fixed", inset: 0, zIndex: 0, pointerEvents: "none" }}
      >
        <Lattice />
      </div>

      <div style={{ position: "relative", zIndex: 1 }}>
        <Nav />
        <Hero onSubmit={onSubmit} />

        {results && !results.pending && results.candidates.length > 0 && (
          <div ref={resultsRef}>
            <Results
              seed={results.seed}
              candidates={results.candidates}
              batchMetrics={results.batchMetrics}
              onClose={() => setResults(null)}
            />
          </div>
        )}

        <HowItWorks />
        <Footer />
      </div>

      {generating && results && (
        <GeneratingOverlay
          seed={results.seed}
          complete={results.candidates.length > 0 || !!results.error}
          error={results.error}
          onDone={onGenDone}
        />
      )}
    </div>
  );
}
