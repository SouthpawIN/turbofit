// Render smoke: execute the shipped bundle against a stub SDK and walk the tree.
// Proves the cluster builds without throwing and emits the dials + gears.
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const source = fs.readFileSync(path.join(root, "dashboard/dist/index.js"), "utf8");

function h(type, props, ...children) {
  return { type, props: props || {}, children: children.flat(Infinity).filter((c) => c != null && c !== false) };
}
const stateValues = [];
let stateIndex = 0;
const hooks = {
  useState(initial) {
    const i = stateIndex++;
    if (stateValues[i] === undefined) stateValues[i] = initial;
    return [stateValues[i], (v) => { stateValues[i] = v; }];
  },
  useEffect() {},
  useCallback(fn) { return fn; },
  useMemo(fn) { return fn(); },
};
const passthrough = (name) => (props) => h(name, props, props && props.children);
const components = {};
for (const name of ["Badge","Button","Card","CardContent","CardHeader","CardTitle","Input","Label","Select","SelectOption","Separator"]) {
  components[name] = passthrough(name);
}
let registered = null;
global.window = {
  __HERMES_PLUGIN_SDK__: {
    React: { createElement: h },
    hooks,
    components,
    fetchJSON: async () => ({}),
  },
  __HERMES_PLUGINS__: { register: (id, fn) => { registered = fn; } },
};

// The bundle is an IIFE reading window.*; run it in this context.
new Function("window", source)(global.window);
if (!registered) throw new Error("bundle did not register a page");

// Seed status with dials so the cluster has something to draw.
const dials = {
  memory_pressure: { ratio: 0.42, state: "comfortable", measured: true },
  context: { current: 65536, native_max: 131072, floor: 65536 },
  chain: {
    label: "Dedicated VRAM",
    levels: [
      { id: "d0", kind: "model", model: "qwen3-8-27b", context: 131072, needs_mb_per_card: 20000, is_active: true, fits_now: true },
      { id: "d1", kind: "model", model: "glm-5-3-flash", context: 131072, needs_mb_per_card: 0, is_active: false, fits_now: true, remote: true, endpoint: "http://100.68.42.68:8091/v1", peer: "omarchy" },
      { id: "d2", kind: "yield", model: null, context: 0, needs_mb_per_card: 0, is_active: false, fits_now: true },
    ],
  },
};
// status is the first useState slot; inject before render.
stateValues[0] = { dials, usage: { tps: 33.26 }, provider: {}, selection: {} };
stateIndex = 0;
const tree = registered();

let gauges = 0, gears = 0, shifts = 0, tailnet = 0;
(function walk(node) {
  if (node == null || typeof node !== "object") return;
  if (Array.isArray(node)) return node.forEach(walk);
  // Function components must be invoked, exactly as React would.
  if (typeof node.type === "function") {
    const saved = stateIndex;
    stateIndex = 0;
    const out = node.type(Object.assign({}, node.props, { children: node.children }));
    stateIndex = saved;
    return walk(out);
  }
  const cls = (node.props && node.props.className) || "";
  if (typeof cls === "string") {
    if (cls.split(" ").includes("tf-gauge")) gauges++;
    if (cls.includes("tf-gear-row")) gears++;
  }
  for (const child of node.children || []) {
    if (typeof child === "string" && child === "tailnet") tailnet++;
    walk(child);
  }
  if (node.props && typeof node.props.onClick === "function") shifts++;
})(tree);

console.log(JSON.stringify({ gauges, gears, clickable: shifts > 0, tailnetBadge: tailnet > 0 }));
if (gauges !== 3) throw new Error("expected 3 dials, got " + gauges);
if (gears !== 3) throw new Error("expected 3 gears, got " + gears);
if (!shifts) throw new Error("no shift control rendered");
console.log("RENDER OK");
