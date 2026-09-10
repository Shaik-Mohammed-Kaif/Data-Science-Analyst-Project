const pptxgen = require("pptxgenjs");

const NAVY = "0D1B2A";
const TEAL = "1B4B43";
const MINT = "00E5A0";
const WHITE = "FFFFFF";
const OFFWHITE = "F4F6F5";
const GRAY = "5B6B66";
const LIGHTGRAY = "9AA8A4";

let pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5

const FONT_HEAD = "Cambria";
const FONT_BODY = "Calibri";

function addFooter(slide, pageNum) {
  slide.addText(`Global EV Market Analytics  |  Real Scraped Data`, {
    x: 0.5, y: 7.15, w: 8, h: 0.3, fontFace: FONT_BODY, fontSize: 9, color: LIGHTGRAY,
  });
  slide.addText(`${pageNum}`, {
    x: 12.6, y: 7.15, w: 0.5, h: 0.3, fontFace: FONT_BODY, fontSize: 9, color: LIGHTGRAY, align: "right",
  });
}

// ============================================================
// SLIDE 1 — Title
// ============================================================
{
  const slide = pres.addSlide();
  slide.background = { color: NAVY };

  slide.addShape(pres.ShapeType.ellipse, {
    x: 10.3, y: -1.5, w: 5.5, h: 5.5, fill: { color: TEAL, transparency: 40 }, line: { type: "none" },
  });
  slide.addShape(pres.ShapeType.ellipse, {
    x: 11.8, y: 4.8, w: 3.2, h: 3.2, fill: { color: MINT, transparency: 75 }, line: { type: "none" },
  });

  slide.addText("GLOBAL EV MARKET ANALYTICS", {
    x: 0.7, y: 2.55, w: 10.5, h: 1.1, fontFace: FONT_HEAD, fontSize: 40, bold: true, color: WHITE,
  });
  slide.addText("A Real, Web-Scraped Data Analysis of the Production Electric Vehicle Market", {
    x: 0.7, y: 3.55, w: 9.5, h: 0.6, fontFace: FONT_BODY, fontSize: 17, italic: true, color: MINT,
  });

  slide.addShape(pres.ShapeType.rect, {
    x: 0.7, y: 4.35, w: 0.55, h: 0.04, fill: { color: MINT }, line: { type: "none" },
  });

  slide.addText([
    { text: "Data Analytics Phase  |  requests + BeautifulSoup  →  Pandas  →  Streamlit\n", options: { color: OFFWHITE, fontSize: 13 } },
    { text: "Source: Wikipedia — List of Battery Electric Vehicles (243 real production models)\n", options: { color: LIGHTGRAY, fontSize: 12 } },
    { text: "Prepared by S Mohammed Kaif", options: { color: LIGHTGRAY, fontSize: 12 } },
  ], { x: 0.7, y: 4.65, w: 8.5, h: 1.4, fontFace: FONT_BODY, lineSpacingMultiple: 1.3 });
}

// ============================================================
// SLIDE 2 — Executive Summary (stat callouts)
// ============================================================
{
  const slide = pres.addSlide();
  slide.background = { color: WHITE };
  slide.addText("Executive Summary", { x: 0.6, y: 0.4, w: 8, h: 0.6, fontFace: FONT_HEAD, fontSize: 30, bold: true, color: NAVY });
  slide.addText("Real production-EV data, scraped and analyzed end to end", {
    x: 0.6, y: 1.0, w: 9, h: 0.4, fontFace: FONT_BODY, fontSize: 14, italic: true, color: GRAY,
  });

  const stats = [
    { n: "243", l: "Real EV Models\nAnalyzed" },
    { n: "59", l: "Manufacturers\nTracked" },
    { n: "17", l: "Origin\nCountries" },
    { n: "2010–26", l: "Launch Years\nCovered" },
  ];
  const boxW = 2.85, gap = 0.25, startX = 0.6, y = 1.7;
  stats.forEach((s, i) => {
    const x = startX + i * (boxW + gap);
    slide.addShape(pres.ShapeType.roundRect, {
      x, y, w: boxW, h: 1.9, rectRadius: 0.08, fill: { color: OFFWHITE }, line: { type: "none" },
      shadow: { type: "outer", color: "999999", opacity: 0.25, blur: 6, offset: 2, angle: 90 },
    });
    slide.addText(s.n, { x, y: y + 0.18, w: boxW, h: 0.85, align: "center", fontFace: FONT_HEAD, fontSize: 40, bold: true, color: TEAL });
    slide.addText(s.l, { x, y: y + 1.05, w: boxW, h: 0.7, align: "center", fontFace: FONT_BODY, fontSize: 12.5, color: GRAY });
  });

  slide.addText("Top-line findings:", { x: 0.6, y: 4.05, w: 6, h: 0.4, fontFace: FONT_BODY, fontSize: 15, bold: true, color: NAVY });

  const bullets = [
    "EV launches accelerated sharply post-2020 — 132 of 243 tracked models launched 2023 onward.",
    "Crossover SUV dominates body styles (140/243 models, 58%), and its share is still rising.",
    "Dedicated BEV-platform adoption jumped from 58% (2020) to 84% (2025) — automakers are moving off converted ICE platforms.",
    "Chinese-market EVs now account for over a third (88/243) of tracked production models.",
  ];
  slide.addText(
    bullets.map(b => ({ text: b, options: { bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 10 } })),
    { x: 0.6, y: 4.5, w: 11.8, h: 2.3, fontFace: FONT_BODY, fontSize: 14, color: "222222", lineSpacingMultiple: 1.15 }
  );

  addFooter(slide, 2);
}

// ============================================================
// SLIDE 3 — Methodology
// ============================================================
{
  const slide = pres.addSlide();
  slide.background = { color: WHITE };
  slide.addText("Methodology", { x: 0.6, y: 0.4, w: 8, h: 0.6, fontFace: FONT_HEAD, fontSize: 30, bold: true, color: NAVY });
  slide.addText("From raw HTML to production-ready insight — a fully reproducible pipeline", {
    x: 0.6, y: 1.0, w: 10, h: 0.4, fontFace: FONT_BODY, fontSize: 14, italic: true, color: GRAY,
  });

  const steps = [
    { n: "01", t: "Scrape", d: "requests + BeautifulSoup pull structured wikitables directly from Wikipedia's production EV listings." },
    { n: "02", t: "Load & Combine", d: "Global-market and Chinese-market tables merged into one dataset with a Market flag." },
    { n: "03", t: "Clean & Inspect", d: "Deduplication, text standardization, missing-value audit, multi-value body-style splitting." },
    { n: "04", t: "Feature Engineer", d: "Model age, launch era, region grouping, manufacturer parent-group roll-up, SUV flag." },
    { n: "05", t: "Statistical Testing", d: "Chi-square, linear regression, and two-proportion z-tests validate every visual pattern." },
    { n: "06", t: "Dashboard + Deck", d: "Interactive Streamlit app for live exploration; this deck for executive presentation." },
  ];

  const colW = 3.85, rowH = 1.55, gapX = 0.25, gapY = 0.25;
  steps.forEach((s, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.6 + col * (colW + gapX);
    const y = 1.75 + row * (rowH + gapY);
    slide.addShape(pres.ShapeType.roundRect, {
      x, y, w: colW, h: rowH, rectRadius: 0.07, fill: { color: OFFWHITE }, line: { type: "none" },
    });
    slide.addShape(pres.ShapeType.ellipse, {
      x: x + 0.2, y: y + 0.2, w: 0.55, h: 0.55, fill: { color: TEAL }, line: { type: "none" },
    });
    slide.addText(s.n, { x: x + 0.2, y: y + 0.2, w: 0.55, h: 0.55, align: "center", valign: "middle", fontFace: FONT_HEAD, fontSize: 16, bold: true, color: WHITE });
    slide.addText(s.t, { x: x + 0.9, y: y + 0.15, w: colW - 1.1, h: 0.35, fontFace: FONT_BODY, fontSize: 14, bold: true, color: NAVY });
    slide.addText(s.d, { x: x + 0.2, y: y + 0.75, w: colW - 0.4, h: 0.7, fontFace: FONT_BODY, fontSize: 10.5, color: GRAY, lineSpacingMultiple: 1.05 });
  });

  addFooter(slide, 3);
}

// ============================================================
// SLIDE 4 — EV Launch Growth (native chart)
// ============================================================
{
  const slide = pres.addSlide();
  slide.background = { color: WHITE };
  slide.addText("Finding 1 — EV Launches Are Accelerating", { x: 0.6, y: 0.4, w: 11, h: 0.6, fontFace: FONT_HEAD, fontSize: 27, bold: true, color: NAVY });
  slide.addText("Statistically significant upward trend confirmed via linear regression (p < 0.05)", {
    x: 0.6, y: 1.0, w: 11, h: 0.4, fontFace: FONT_BODY, fontSize: 13, italic: true, color: GRAY,
  });

  const years = ["2010","2015","2016","2017","2018","2019","2020","2021","2022","2023","2024","2025"];
  const launches = [1,1,1,1,2,12,25,30,38,53,50,24];

  slide.addChart(pres.ChartType.bar, [
    { name: "Models Launched", labels: years, values: launches },
  ], {
    x: 0.6, y: 1.6, w: 8.3, h: 5.2,
    chartColors: [TEAL],
    showTitle: false, showLegend: false,
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: NAVY, dataLabelFontSize: 10,
    catAxisLabelColor: GRAY, valAxisLabelColor: GRAY,
    valGridLine: { color: "E5E5E5", size: 1 }, catGridLine: { style: "none" },
    barGapWidthPct: 30,
  });

  slide.addShape(pres.ShapeType.roundRect, {
    x: 9.15, y: 1.7, w: 3.55, h: 5.0, rectRadius: 0.08, fill: { color: NAVY }, line: { type: "none" },
  });
  slide.addText("53", { x: 9.15, y: 2.0, w: 3.55, h: 1.0, align: "center", fontFace: FONT_HEAD, fontSize: 48, bold: true, color: MINT });
  slide.addText("models launched in 2023 alone — the single biggest year on record", {
    x: 9.45, y: 3.0, w: 2.95, h: 1.1, align: "center", fontFace: FONT_BODY, fontSize: 12, color: WHITE,
  });
  slide.addShape(pres.ShapeType.rect, { x: 9.65, y: 4.25, w: 2.55, h: 0.02, fill: { color: MINT }, line: { type: "none" } });
  slide.addText("132 of 243 tracked models (54%) launched in just the last 4 years (2023-2026).", {
    x: 9.45, y: 4.5, w: 2.95, h: 1.9, align: "center", fontFace: FONT_BODY, fontSize: 12, color: OFFWHITE, valign: "top",
  });

  addFooter(slide, 4);
}

// ============================================================
// SLIDE 5 — Market Split & Body Style
// ============================================================
{
  const slide = pres.addSlide();
  slide.background = { color: WHITE };
  slide.addText("Finding 2 — China Is Now a Core Battleground", { x: 0.6, y: 0.4, w: 11.5, h: 0.6, fontFace: FONT_HEAD, fontSize: 27, bold: true, color: NAVY });
  slide.addText("Body-style preference differs significantly by market (chi-square, p < 0.05)", {
    x: 0.6, y: 1.0, w: 11, h: 0.4, fontFace: FONT_BODY, fontSize: 13, italic: true, color: GRAY,
  });

  slide.addChart(pres.ChartType.pie, [
    { name: "Market", labels: ["Global Market", "Chinese Market"], values: [155, 88] },
  ], {
    x: 0.6, y: 1.7, w: 5.6, h: 5.0,
    chartColors: [TEAL, MINT],
    showTitle: true, title: "Model Count by Market", titleFontSize: 14, titleColor: NAVY,
    showLegend: true, legendPos: "b", legendColor: GRAY,
    showValue: true, dataLabelColor: WHITE, dataLabelFontSize: 12,
  });

  const bodyLabels = ["Crossover SUV", "Sedan", "Hatchback", "Liftback", "MPV/Minivan"];
  const bodyValues = [140, 41, 29, 10, 5];
  slide.addChart(pres.ChartType.bar, [
    { name: "Models", labels: bodyLabels, values: bodyValues },
  ], {
    x: 6.6, y: 1.7, w: 6.1, h: 5.0,
    barDir: "bar",
    chartColors: [NAVY],
    showTitle: true, title: "Top Body Styles (All Markets)", titleFontSize: 14, titleColor: NAVY,
    showLegend: false, showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelColor: NAVY,
    catAxisLabelColor: GRAY, valAxisLabelColor: GRAY,
    valGridLine: { color: "E5E5E5", size: 1 }, catGridLine: { style: "none" },
  });

  addFooter(slide, 5);
}

// ============================================================
// SLIDE 6 — Manufacturer Landscape
// ============================================================
{
  const slide = pres.addSlide();
  slide.background = { color: WHITE };
  slide.addText("Finding 3 — The Manufacturer Landscape Is Consolidating", {
    x: 0.6, y: 0.4, w: 12, h: 0.6, fontFace: FONT_HEAD, fontSize: 25, bold: true, color: NAVY,
  });
  slide.addText("Parent-group roll-up reveals concentration behind dozens of individual brand names", {
    x: 0.6, y: 1.0, w: 11, h: 0.4, fontFace: FONT_BODY, fontSize: 13, italic: true, color: GRAY,
  });

  const groups = ["Stellantis","Geely Group","General Motors","BYD Auto","Volkswagen Group","BMW Group"];
  const counts = [31,20,15,14,13,12];
  slide.addChart(pres.ChartType.bar, [
    { name: "Models", labels: groups, values: counts },
  ], {
    x: 0.6, y: 1.7, w: 11.9, h: 4.7,
    chartColors: [TEAL],
    showTitle: false, showLegend: false,
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: NAVY, dataLabelFontSize: 11,
    catAxisLabelColor: GRAY, valAxisLabelColor: GRAY, catAxisLabelFontSize: 11,
    valGridLine: { color: "E5E5E5", size: 1 }, catGridLine: { style: "none" },
    barGapWidthPct: 40,
  });

  addFooter(slide, 6);
}

// ============================================================
// SLIDE 7 — Platform Strategy Shift
// ============================================================
{
  const slide = pres.addSlide();
  slide.background = { color: WHITE };
  slide.addText("Finding 4 — Automakers Are Betting on Purpose-Built EVs", {
    x: 0.6, y: 0.4, w: 12, h: 0.6, fontFace: FONT_HEAD, fontSize: 25, bold: true, color: NAVY,
  });
  slide.addText("Dedicated BEV platform adoption and SUV share are both rising together", {
    x: 0.6, y: 1.0, w: 11, h: 0.4, fontFace: FONT_BODY, fontSize: 13, italic: true, color: GRAY,
  });

  // left stat card - dedicated BEV
  slide.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 1.75, w: 5.7, h: 4.85, rectRadius: 0.08, fill: { color: OFFWHITE }, line: { type: "none" } });
  slide.addText("Dedicated BEV Platform Share", { x: 0.9, y: 2.0, w: 5.1, h: 0.4, fontFace: FONT_BODY, fontSize: 14, bold: true, color: NAVY });
  slide.addText("58%", { x: 0.9, y: 2.5, w: 2.3, h: 1.1, fontFace: FONT_HEAD, fontSize: 44, bold: true, color: GRAY });
  slide.addText("2020", { x: 0.9, y: 3.55, w: 2.3, h: 0.35, align: "left", fontFace: FONT_BODY, fontSize: 12, color: GRAY });
  slide.addText("➜", { x: 3.15, y: 2.75, w: 0.8, h: 0.8, align: "center", fontFace: FONT_BODY, fontSize: 28, color: MINT });
  slide.addText("84%", { x: 3.9, y: 2.5, w: 2.3, h: 1.1, fontFace: FONT_HEAD, fontSize: 44, bold: true, color: TEAL });
  slide.addText("2025", { x: 3.9, y: 3.55, w: 2.3, h: 0.35, align: "left", fontFace: FONT_BODY, fontSize: 12, color: GRAY });
  slide.addText("Purpose-built EV architecture is fast becoming the industry default rather than the exception — converted ICE platforms are on the way out.", {
    x: 0.9, y: 4.35, w: 5.1, h: 2.0, fontFace: FONT_BODY, fontSize: 12.5, color: "222222", lineSpacingMultiple: 1.2,
  });

  // right stat card - SUV share
  slide.addShape(pres.ShapeType.roundRect, { x: 6.55, y: 1.75, w: 5.9, h: 4.85, rectRadius: 0.08, fill: { color: NAVY }, line: { type: "none" } });
  slide.addText("SUV/Crossover Body-Style Share", { x: 6.85, y: 2.0, w: 5.3, h: 0.4, fontFace: FONT_BODY, fontSize: 14, bold: true, color: WHITE });
  slide.addText("44%", { x: 6.85, y: 2.5, w: 2.3, h: 1.1, fontFace: FONT_HEAD, fontSize: 44, bold: true, color: LIGHTGRAY });
  slide.addText("2010-19", { x: 6.85, y: 3.55, w: 2.3, h: 0.35, align: "left", fontFace: FONT_BODY, fontSize: 12, color: LIGHTGRAY });
  slide.addText("➜", { x: 9.1, y: 2.75, w: 0.8, h: 0.8, align: "center", fontFace: FONT_BODY, fontSize: 28, color: MINT });
  slide.addText("67%", { x: 9.85, y: 2.5, w: 2.3, h: 1.1, fontFace: FONT_HEAD, fontSize: 44, bold: true, color: MINT });
  slide.addText("2023-26", { x: 9.85, y: 3.55, w: 2.3, h: 0.35, align: "left", fontFace: FONT_BODY, fontSize: 12, color: LIGHTGRAY });
  slide.addText("Two-proportion z-test confirms this shift is statistically significant, not random noise — SUVs are the new EV default body style.", {
    x: 6.85, y: 4.35, w: 5.3, h: 2.0, fontFace: FONT_BODY, fontSize: 12.5, color: OFFWHITE, lineSpacingMultiple: 1.2,
  });

  addFooter(slide, 7);
}

// ============================================================
// SLIDE 8 — Business Recommendations
// ============================================================
{
  const slide = pres.addSlide();
  slide.background = { color: WHITE };
  slide.addText("Business Recommendations", { x: 0.6, y: 0.4, w: 10, h: 0.6, fontFace: FONT_HEAD, fontSize: 30, bold: true, color: NAVY });
  slide.addText("What these patterns mean for product, market entry, and platform strategy", {
    x: 0.6, y: 1.0, w: 11, h: 0.4, fontFace: FONT_BODY, fontSize: 13, italic: true, color: GRAY,
  });

  const recs = [
    { t: "Prioritize dedicated BEV platforms", d: "84% of 2025 launches use purpose-built architecture — converted-ICE EVs risk looking dated to buyers and dealers alike." },
    { t: "Lead with Crossover SUV body styles", d: "58% of all tracked models (and rising) are Crossover SUVs — this is the safest default body style for a new EV entry." },
    { t: "Treat China as a distinct product market", d: "Body-style mix differs significantly between Global and Chinese markets — a one-size-fits-all model lineup will underperform." },
    { t: "Expect continued launch-volume growth", d: "The 2010-2025 launch trend is statistically significant and upward — competitive intensity will keep increasing, not plateau." },
  ];

  const colW = 5.75, rowH = 2.25, gapX = 0.3, gapY = 0.25;
  recs.forEach((r, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.6 + col * (colW + gapX);
    const y = 1.75 + row * (rowH + gapY);
    slide.addShape(pres.ShapeType.roundRect, { x, y, w: colW, h: rowH, rectRadius: 0.08, fill: { color: OFFWHITE }, line: { type: "none" } });
    slide.addShape(pres.ShapeType.rect, { x: x + 0.35, y: y + 0.3, w: 0.5, h: 0.5, fill: { color: MINT }, line: { type: "none" } });
    slide.addText(`${i+1}`, { x: x + 0.35, y: y + 0.3, w: 0.5, h: 0.5, align: "center", valign: "middle", fontFace: FONT_HEAD, fontSize: 16, bold: true, color: NAVY });
    slide.addText(r.t, { x: x + 1.05, y: y + 0.28, w: colW - 1.35, h: 0.55, fontFace: FONT_BODY, fontSize: 15, bold: true, color: NAVY });
    slide.addText(r.d, { x: x + 0.35, y: y + 0.95, w: colW - 0.7, h: 1.15, fontFace: FONT_BODY, fontSize: 12, color: "333333", lineSpacingMultiple: 1.15 });
  });

  addFooter(slide, 8);
}

// ============================================================
// SLIDE 9 — What's Next (roadmap)
// ============================================================
{
  const slide = pres.addSlide();
  slide.background = { color: NAVY };
  slide.addText("What's Next in This Project", { x: 0.6, y: 0.6, w: 10, h: 0.7, fontFace: FONT_HEAD, fontSize: 30, bold: true, color: WHITE });
  slide.addText("This is Phase 1 of a 3-phase build across Data Analytics, Data Science, and Machine Learning", {
    x: 0.6, y: 1.3, w: 11.5, h: 0.4, fontFace: FONT_BODY, fontSize: 13, italic: true, color: LIGHTGRAY,
  });

  const phases = [
    { p: "Phase 1 ✓", t: "Data Analytics", d: "Scraping, cleaning, feature engineering, statistical testing, and this dashboard + deck.", done: true },
    { p: "Phase 2", t: "Data Science", d: "Predictive modeling, clustering, and deeper explainability on the EV market dataset.", done: false },
    { p: "Phase 3", t: "Machine Learning", d: "Production model packaged behind a live Streamlit prediction web app.", done: false },
  ];

  const colW = 3.85, gapX = 0.25, y = 2.3;
  phases.forEach((ph, i) => {
    const x = 0.6 + i * (colW + gapX);
    slide.addShape(pres.ShapeType.roundRect, {
      x, y, w: colW, h: 3.6, rectRadius: 0.08,
      fill: { color: ph.done ? MINT : TEAL }, line: { type: "none" },
    });
    slide.addText(ph.p, { x: x + 0.25, y: y + 0.25, w: colW - 0.5, h: 0.4, fontFace: FONT_BODY, fontSize: 12, bold: true, color: ph.done ? NAVY : OFFWHITE });
    slide.addText(ph.t, { x: x + 0.25, y: y + 0.65, w: colW - 0.5, h: 0.55, fontFace: FONT_HEAD, fontSize: 20, bold: true, color: ph.done ? NAVY : WHITE });
    slide.addText(ph.d, { x: x + 0.25, y: y + 1.3, w: colW - 0.5, h: 2.1, fontFace: FONT_BODY, fontSize: 12.5, color: ph.done ? "1B4B43" : OFFWHITE, lineSpacingMultiple: 1.2 });
  });

  slide.addText("S Mohammed Kaif  |  mohammedkaif8297@gmail.com", {
    x: 0.6, y: 6.6, w: 8, h: 0.4, fontFace: FONT_BODY, fontSize: 11, color: LIGHTGRAY,
  });
}

pres.writeFile({ fileName: "/home/claude/ev_project_v2/presentation/EV_Market_Business_Insights.pptx" })
  .then(() => console.log("Presentation saved"));
