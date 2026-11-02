// Generates header banner for PYRO (1200x420)
import { writeFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";

const OUT = resolve("D:/quakequack/PYRO/assets");
mkdirSync(OUT, { recursive: true });

const MONO = `'Consolas','JetBrains Mono',monospace`;
const DISP = `'Arial Black',Impact,sans-serif`;
const SANS = `'Segoe UI',system-ui,sans-serif`;

const INK = "#0a0f1b";
const ORANGE = "#ff7a1a";
const RED = "#e8381b";
const YELLOW = "#ffd23f";
const CYAN = "#00d0ff";

function pyroBanner() {
  const W = 1200, H = 400;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#0a0f18"/>
    <stop offset="0.6" stop-color="#140a08"/>
    <stop offset="1" stop-color="#240c06"/>
  </linearGradient>
  <radialGradient id="glow" cx="0.8" cy="0.5" r="0.6">
    <stop offset="0" stop-color="#ff5a1f" stop-opacity="0.35"/>
    <stop offset="0.6" stop-color="#ff9f45" stop-opacity="0.10"/>
    <stop offset="1" stop-color="#000000" stop-opacity="0"/>
  </radialGradient>
  <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
    <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#ffffff" stroke-opacity="0.04" stroke-width="1"/>
  </pattern>
  <style>
    .mono { font-family: ${MONO}; font-weight: 700; }
    .disp { font-family: ${DISP}; font-weight: 900; }
    .sans { font-family: ${SANS}; }
    .blink { animation: blink 1.2s steps(1,end) infinite; }
    @keyframes blink { 50% { opacity: 0; } }
  </style>
</defs>

<!-- BG -->
<rect width="${W}" height="${H}" fill="url(#bg)"/>
<rect width="${W}" height="${H}" fill="url(#grid)"/>
<ellipse cx="960" cy="200" rx="360" ry="240" fill="url(#glow)"/>

<!-- Top Status Line -->
<g transform="translate(60, 48)">
  <rect x="0" y="0" width="130" height="26" rx="6" fill="#ff5a1f" fill-opacity="0.18" stroke="#ff5a1f" stroke-width="1.5"/>
  <circle cx="14" cy="13" r="4" fill="#ff5a1f" class="blink"/>
  <text x="26" y="17" class="mono" font-size="11" letter-spacing="1.5" fill="#ff9f45">UAV INFERENCE</text>

  <text x="146" y="17" class="mono" font-size="11" letter-spacing="1.5" fill="#8a9ba8">// FLAME AERIAL BENCHMARK &#183; 384&#215;384 REAL-TIME</text>
</g>

<!-- Big Title -->
<g transform="translate(56, 150)">
  <text class="disp" font-size="94" letter-spacing="-1" fill="#ff5a1f" opacity="0.3" x="4" y="4">PYRO</text>
  <text class="disp" font-size="94" letter-spacing="-1" fill="#ffffff">PYRO</text>
  <text class="mono" font-size="18" letter-spacing="3" fill="#ff9f45" x="310" y="-30">v1.0</text>
</g>

<!-- Subtitle & Specs -->
<g transform="translate(60, 205)">
  <text class="sans" font-size="20" font-weight="600" fill="#e2e8f0">Pixel-Level pYRotechnic Outlining</text>
  <text class="sans" font-size="15" fill="#94a3b8" y="28">Lightweight Semantic Fire Segmentation for Autonomous UAVs &amp; Edge Drones</text>

  <!-- Spec Pills -->
  <g transform="translate(0, 75)">
    <!-- Pill 1 -->
    <rect x="0" y="0" width="145" height="32" rx="7" fill="#ffffff" fill-opacity="0.06" stroke="#ffffff" stroke-opacity="0.15" stroke-width="1.2"/>
    <text x="72" y="20" text-anchor="middle" class="mono" font-size="11" fill="#f8fafc">MobileNetV3</text>

    <!-- Pill 2 -->
    <rect x="157" y="0" width="155" height="32" rx="7" fill="#ffffff" fill-opacity="0.06" stroke="#ffffff" stroke-opacity="0.15" stroke-width="1.2"/>
    <text x="234" y="20" text-anchor="middle" class="mono" font-size="11" fill="#f8fafc">ASPP Bottleneck</text>

    <!-- Pill 3 -->
    <rect x="324" y="0" width="170" height="32" rx="7" fill="#ffffff" fill-opacity="0.06" stroke="#ffffff" stroke-opacity="0.15" stroke-width="1.2"/>
    <text x="409" y="20" text-anchor="middle" class="mono" font-size="11" fill="#f8fafc">5-Stage SE U-Net</text>

    <!-- Pill 4 -->
    <rect x="506" y="0" width="175" height="32" rx="7" fill="#ff5a1f" fill-opacity="0.15" stroke="#ff5a1f" stroke-width="1.2"/>
    <text x="593" y="20" text-anchor="middle" class="mono" font-size="11" fill="#ff9f45">Tversky + Focal</text>
  </g>

  <!-- Metric highlight -->
  <text x="0" y="145" class="mono" font-size="12" letter-spacing="1.5" fill="#64748b">SURVIVES FIRE PIXELS &lt; 1% OF IMAGE &#183; TRAINED PYTORCH CHECKPOINT INCLUDED</text>
</g>

<!-- Right Side: Graphic Visualizer -->
<g transform="translate(850, 75)">
  <!-- Outer Box -->
  <rect x="0" y="0" width="290" height="260" rx="14" fill="#0c121e" stroke="#ffffff" stroke-opacity="0.12" stroke-width="2"/>
  
  <!-- Target Grid -->
  <g stroke="#ff5a1f" stroke-opacity="0.25" stroke-width="1" stroke-dasharray="4 4">
    <path d="M 0 65 H 290 M 0 130 H 290 M 0 195 H 290"/>
    <path d="M 72 0 V 260 M 145 0 V 260 M 217 0 V 260"/>
  </g>

  <!-- Detection Bounding Box -->
  <rect x="55" y="45" width="175" height="155" rx="8" fill="#ff5a1f" fill-opacity="0.12" stroke="#ff5a1f" stroke-width="2"/>
  <rect x="55" y="45" width="95" height="22" rx="4" fill="#ff5a1f"/>
  <text x="62" y="60" class="mono" font-size="10" font-weight="700" fill="#ffffff">FIRE: 0.984</text>

  <!-- Contour Simulation -->
  <path d="M 85 175 Q 110 95 145 75 Q 175 110 195 145 Q 165 185 125 180 Z" fill="#ff9f45" fill-opacity="0.35" stroke="#ffd23f" stroke-width="2"/>

  <!-- Crosshairs -->
  <path d="M 145 20 V 40 M 145 220 V 240 M 20 130 H 40 M 250 130 H 270" stroke="#00d0ff" stroke-width="2"/>

  <!-- Footer Tag -->
  <rect x="15" y="215" width="260" height="30" rx="6" fill="#000000" fill-opacity="0.5"/>
  <text x="25" y="234" class="mono" font-size="10" fill="#94a3b8">IoU: <tspan fill="#00d0ff">0.912</tspan> &#183; LATENCY: <tspan fill="#34d399">18ms</tspan></text>
</g>

<!-- Accent Lines -->
<rect x="0" y="396" width="${W}" height="4" fill="#ff5a1f"/>
</svg>`;
}

writeFileSync(resolve(OUT, "pyro-banner.svg"), pyroBanner(), "utf8");
console.log("PYRO banner created!");
