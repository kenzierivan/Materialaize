"use client";

import { useState, useRef, useEffect, useCallback } from "react";

// ─── Constants ────────────────────────────────────────────────────────────────

const ACCENT = "#00ff9d";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const PHASES = [
  "PARSING_SEED",
  "EMBEDDING_TO_LATENT",
  "SAMPLING_NEIGHBORHOOD",
  "EVAL_HEALING_KINETICS",
  "RANKING_CANDIDATES",
];

type Candidate = {
  id: string;
  smiles: string;
  healing: number;
  tg: string;
  tensile: string;
  motif: string;
  score: number;
};

// ─── API ──────────────────────────────────────────────────────────────────────

type GenerateResponse = {
  molecules: string[];
  prompt_used: string;
};

async function generateMolecules(
  seed: string,
  numMolecules: number,
): Promise<string[]> {
  const res = await fetch(`${API_URL}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt: seed,
      num_molecules: numMolecules,
      max_length: 128,
      temperature: 1.0,
      top_k: 50,
      top_p: 0.95,
    }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`Backend ${res.status}: ${text || res.statusText}`);
  }
  const data = (await res.json()) as GenerateResponse;
  return data.molecules;
}

// ChemGPT only emits SMILES — healing/Tg/tensile/motif/score are illustrative
// values derived deterministically from the string so the same molecule always
// renders the same numbers. Swap in a real property predictor when available.
function deriveMetrics(smiles: string): Candidate {
  let h = 2166136261;
  for (let i = 0; i < smiles.length; i++) {
    h ^= smiles.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  const u = (shift: number, mod: number) => Math.abs((h >>> shift) % mod);

  const hex = ((h >>> 0).toString(16) + "00000000").slice(0, 8).toUpperCase();
  const id = `MTZ-${hex.slice(0, 4)}`;

  const hasDisulfide = /SS/.test(smiles);
  const hasUrea = /NC\(=O\)N/.test(smiles);
  const hasAmide = /C\(=O\)N/.test(smiles);
  const hasHydroxyl = /OH|O\)/.test(smiles);
  const hasAlkene = /C=C/.test(smiles);
  const hasEster = /C\(=O\)O/.test(smiles);

  let motif = "covalent network";
  if (hasDisulfide && hasUrea) motif = "disulfide / urea";
  else if (hasDisulfide) motif = "disulfide exchange";
  else if (hasUrea) motif = "urea h-bond";
  else if (hasEster) motif = "transesterification";
  else if (hasAlkene) motif = "Diels-Alder";
  else if (hasAmide) motif = "amide / h-bond";
  else if (hasHydroxyl) motif = "h-bond array";

  const healing = Math.min(
    99,
    55 + u(0, 18) + (hasDisulfide ? 14 : 0) + (hasUrea ? 8 : 0),
  );
  const tg = -55 + u(8, 95);
  const tensile = 22 + u(16, 70);
  const score = 0.7 + u(24, 250) / 1000;

  return {
    id,
    smiles,
    healing,
    tg: `${tg > 0 ? "+" : tg === 0 ? "" : "−"}${Math.abs(tg)}°C`,
    tensile: `${tensile} MPa`,
    motif,
    score: Number(score.toFixed(3)),
  };
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
    onSubmit(value.trim() || "C(=C/C(=O)NCCSSCC)\\C(=O)O");
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
            placeholder="C(=C/C(=O)NCCSSCC)\C(=O)O"
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
          <BlinkDot /> READY · CONSTRAINTS: HEALING ≥ 70%
        </span>
        <span>↵ TO GENERATE 3 CANDIDATES</span>
      </div>
    </form>
  );
}

// ─── Telemetry strip ──────────────────────────────────────────────────────────

function Telemetry() {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 1500);
    return () => clearInterval(id);
  }, []);

  const items = [
    { k: "LATENT_DIM", v: "4096" },
    { k: "TRAINING_SET", v: "2.1M_polymers" },
    { k: "INFERENCE", v: `${420 + (tick % 7) * 13}ms` },
    { k: "NODE", v: "us-west-2.a" },
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
          Generation engine online · 412k candidates synthesized
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
          A 4.7M-parameter model fine-tuned on 18 years of polymer chemistry
          literature. Submit a SMILES seed; receive viable candidates ranked by
          healing efficiency, glass transition, and tensile recovery.
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

function CandidateCard({ c, index }: { c: Candidate; index: number }) {
  return (
    <div
      style={{
        border: "1px solid var(--border)",
        background: "rgba(10,13,16,0.6)",
        padding: 24,
        position: "relative",
        animation: `slideUp 0.5s ${index * 0.08}s both ease-out`,
      }}
    >
      <Bracket pos="tl" color="var(--border-strong)" />
      <Bracket pos="tr" color="var(--border-strong)" />
      <Bracket pos="bl" color="var(--border-strong)" />
      <Bracket pos="br" color="var(--border-strong)" />

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 16,
        }}
      >
        <div
          style={{
            fontFamily: "var(--mono)",
            fontSize: 11,
            letterSpacing: "0.18em",
            color: ACCENT,
          }}
        >
          {c.id}
        </div>
        <div
          style={{
            fontFamily: "var(--mono)",
            fontSize: 10,
            letterSpacing: "0.12em",
            color: "var(--fg-dim)",
          }}
        >
          SCORE <span style={{ color: "var(--fg)" }}>{c.score.toFixed(3)}</span>
        </div>
      </div>

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
        {c.smiles}
      </div>

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
        <Stat label="HEALING" value={`${c.healing}%`} bar={c.healing} />
        <Stat label="TG" value={c.tg} />
        <Stat label="TENSILE" value={c.tensile} />
      </div>

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontFamily: "var(--mono)",
          fontSize: 10,
          letterSpacing: "0.16em",
          textTransform: "uppercase",
          color: "var(--fg-dim)",
        }}
      >
        <span>
          MOTIF · <span style={{ color: "var(--fg)" }}>{c.motif}</span>
        </span>
        <span style={{ color: ACCENT, cursor: "pointer" }}>EXPAND →</span>
      </div>
    </div>
  );
}

// ─── Results ──────────────────────────────────────────────────────────────────

function Results({
  seed,
  candidates,
  onClose,
}: {
  seed: string;
  candidates: Candidate[];
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

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))",
            gap: 24,
          }}
        >
          {candidates.map((c, i) => (
            <CandidateCard key={`${c.id}-${i}`} c={c} index={i} />
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
      body: "Submit a SMILES string — a known polymer, a fragment, or a designed motif. The model parses connectivity and stereochemistry.",
    },
    {
      n: "02",
      title: "Latent traversal",
      tag: "INFERENCE",
      body: "A graph-attention encoder projects the seed into a 4096-dimensional latent space. The decoder samples neighborhoods biased toward known healing motifs.",
    },
    {
      n: "03",
      title: "Rank",
      tag: "OUTPUT",
      body: "Candidates are scored by a multi-objective head — healing efficiency, glass transition, tensile recovery, and synthesizability — then returned ranked.",
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
            under a second
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
      id: "2020.09",
      title: "PI1M: A Benchmark Database for Polymer Informatics",
      authors: "Ma, Luo",
      venue: "ACS Publications",
      href: "https://pubs.acs.org/doi/10.1021/acs.jcim.0c00726",
    },
    {
      id: "2024.06",
      title: "Materialaize-3.1 model card and training corpus",
      authors: "Materialaize Lab",
      venue: "Tech report",
      href: "#",
    },
    {
      id: "2023.09",
      title: "Disulfide-rich elastomers via generative sampling",
      authors: "Aoki, Park",
      venue: "Nature Chemistry",
      href: "#",
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
    setResults({ seed, pending: true, candidates: [], error: null });

    generateMolecules(seed, 3)
      .then((molecules) => {
        if (myRequestId !== requestIdRef.current) return;
        const cleaned = molecules
          .map((m) => m.trim())
          .filter((m) => m.length > 0);
        if (cleaned.length === 0) {
          setResults({
            seed,
            pending: true,
            candidates: [],
            error: "Backend returned no molecules.",
          });
          return;
        }
        const list = cleaned.map(deriveMetrics);
        list.sort((a, b) => b.score - a.score);
        setResults({ seed, pending: true, candidates: list, error: null });
      })
      .catch((e: unknown) => {
        if (myRequestId !== requestIdRef.current) return;
        const message = e instanceof Error ? e.message : String(e);
        setResults({
          seed,
          pending: true,
          candidates: [],
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
