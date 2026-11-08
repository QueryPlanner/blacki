// Interpreter for compiled Blacki UI expressions. Walks the IR tree; never evals.

const isNum = (v) => typeof v === "number" && Number.isFinite(v);

function rowsOf(env, q) {
  const result = env.data[q];
  return result && Array.isArray(result.rows) ? result.rows : [];
}

function filtered(node, env) {
  const rows = rowsOf(env, node.q);
  return node.where ? rows.filter((row) => truthy(evaluate(node.where, env, row))) : rows;
}

const truthy = (v) => v === true || (isNum(v) && v !== 0) || (typeof v === "string" && v !== "");

function compare(op, a, b) {
  if (a == null || b == null) return false;
  switch (op) {
    case "<": return a < b;
    case "<=": return a <= b;
    case ">": return a > b;
    case ">=": return a >= b;
    case "==": return a === b;
    case "!=": return a !== b;
    default: return false;
  }
}

function arithmetic(op, a, b) {
  if (op === "+" && (typeof a === "string" || typeof b === "string")) return `${a ?? ""}${b ?? ""}`;
  if (!isNum(a) || !isNum(b)) return null;
  switch (op) {
    case "+": return a + b;
    case "-": return a - b;
    case "*": return a * b;
    case "/": return b === 0 ? null : a / b;
    default: return null;
  }
}

function reduce(node, env) {
  const rows = filtered(node, env);
  if (node.f === "count") return rows.length;
  if (node.f === "percent") {
    const all = rowsOf(env, node.q).length;
    return all ? Math.round((rows.length / all) * 100) : 0;
  }
  const values = rows.map((r) => r[node.field]).filter(isNum);
  if (node.f === "latest") {
    for (let i = rows.length - 1; i >= 0; i -= 1) {
      const v = rows[i][node.field];
      if (v != null && v !== "") return v;
    }
    return null;
  }
  if (!values.length) return null;
  switch (node.f) {
    case "avg": return values.reduce((t, v) => t + v, 0) / values.length;
    case "sum": return values.reduce((t, v) => t + v, 0);
    case "min": return Math.min(...values);
    case "max": return Math.max(...values);
    default: return null;
  }
}

export function evaluate(node, env, row = null) {
  if (!node || typeof node !== "object") return null;
  switch (node.k) {
    case "lit": return node.v;
    case "var": return env.vars[node.n] ?? null;
    case "row": return row ? row[node.f] ?? null : null;
    case "neg": { const v = evaluate(node.a, env, row); return isNum(v) ? -v : null; }
    case "bin": {
      const a = evaluate(node.l, env, row);
      const b = evaluate(node.r, env, row);
      return ["<", "<=", ">", ">=", "==", "!="].includes(node.o) ? compare(node.o, a, b) : arithmetic(node.o, a, b);
    }
    case "fn": {
      if (node.f === "round") {
        const v = evaluate(node.a, env, row);
        if (!isNum(v)) return null;
        const p = 10 ** (node.d || 0);
        return Math.round(v * p) / p;
      }
      if (node.f === "or") {
        for (const arg of node.a) {
          const v = evaluate(arg, env, row);
          if (v != null && v !== "") return v;
        }
        return null;
      }
      return reduce(node, env);
    }
    default: return null;
  }
}

export function formatValue(v) {
  if (v == null || v === "") return "–";
  if (typeof v === "boolean") return v ? "Yes" : "No";
  if (isNum(v)) {
    const rounded = Math.abs(v) >= 100 ? Math.round(v) : Math.round(v * 10) / 10;
    return rounded.toLocaleString("en-GB");
  }
  return String(v);
}

export function initialVars(vars, env) {
  const out = {};
  for (const [name, spec] of Object.entries(vars || {})) {
    let value = spec.value;
    if (spec.init) {
      const computed = evaluate(spec.init, { vars: {}, data: env.data }, null);
      value = computed ?? 0;
    }
    out[name] = value;
  }
  return out;
}
