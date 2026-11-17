"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(
  path.join(__dirname, "..", "webui", "culling", "static", "js", "common.js"),
  "utf8"
);

function createHarness(fetchImpl = async () => { throw new Error("Unexpected fetch"); }) {
  let value = "";
  let scrollTop = 0;
  const box = {
    clientHeight: 100,
    get value() { return value; },
    set value(text) {
      value = text;
      scrollTop = Math.max(0, this.scrollHeight - this.clientHeight);
    },
    get scrollHeight() {
      return Math.max(this.clientHeight, (value ? value.split("\n").length : 1) * 10);
    },
    get scrollTop() { return scrollTop; },
    set scrollTop(top) {
      scrollTop = Math.min(Math.max(0, top), this.scrollHeight - this.clientHeight);
    },
  };
  const dialog = {
    length: 1,
    dialog() { return this; },
    find() { return this; },
    hide() { return this; },
    show() { return this; },
  };
  function $(selector) {
    if (typeof selector === "function") return;
    return dialog;
  }
  $.extend = Object.assign;
  let interval;
  const context = vm.createContext({
    $, document: {
      title: "Test",
      getElementById(id) { return id === "output" ? box : null; },
    },
    console, AbortController, setTimeout, clearTimeout,
    setInterval(callback) { interval = callback; return 1; },
    clearInterval() { interval = null; },
    fetch: fetchImpl,
  });
  vm.runInContext(source, context);
  const LogDialog = vm.runInContext("LogDialog", context);
  const job = LogDialog.create({
    dialogId: "dialog",
    outputId: "output",
    statusUrl: (since) => `status?since=${since}`,
    label: "test",
  });
  return { job, box, tick: () => interval() };
}

function numberedLines(start, count) {
  return Array.from({ length: count }, (_, i) => `line ${start + i}`);
}

test("keeps exactly 1024 lines and trims to the newest 512 on overflow", () => {
  const { job, box } = createHarness();
  job.appendLines(numberedLines(0, 1024));
  assert.equal(box.value.split("\n").length, 1024);
  job.appendLines(["line 1024"]);
  assert.equal(box.value, numberedLines(513, 512).join("\n"));
  job.appendLines(numberedLines(1025, 513));
  assert.equal(box.value, numberedLines(1026, 512).join("\n"));
});

test("handles oversized batches, embedded newlines, and blank lines", () => {
  const { job, box } = createHarness();
  job.appendLines(["old"]);
  const incoming = numberedLines(0, 1500);
  incoming[1000] = "";
  job.appendLines([incoming.join("\n")]);
  assert.equal(box.value, incoming.slice(-512).join("\n"));
  assert.equal(box.value.split("\n").length, 512);
  job.reset();
  job.appendLines(["first\n\nthird", "", "last"]);
  assert.equal(box.value, "first\n\nthird\n\nlast");
});

test("reset clears the text and new entries start a fresh log", () => {
  const { job, box } = createHarness();
  job.open("Running");
  assert.equal(box.value, "Starting test…");
  job.appendLines(numberedLines(0, 1025));
  job.reset();
  assert.equal(box.value, "");
  job.appendLines(["fresh"]);
  assert.equal(box.value, "fresh");
});

test("follows the bottom and preserves the retained viewport when scrolled up", () => {
  const { job, box } = createHarness();
  job.appendLines(numberedLines(0, 1024));
  assert.equal(box.scrollTop, box.scrollHeight - box.clientHeight);
  box.scrollTop = 8000;
  job.appendLines(["line 1024"]);
  assert.equal(box.scrollTop, 2870);
  assert.equal(box.value.split("\n")[287], "line 800");
  box.scrollTop = 0;
  job.appendLines(["line 1025"]);
  assert.equal(box.scrollTop, 0);
  box.scrollTop = box.scrollHeight;
  job.appendLines(numberedLines(1026, 511));
  assert.equal(box.scrollTop, box.scrollHeight - box.clientHeight);
});

test("polling advances its server cursor independently of display trimming", async () => {
  const urls = [];
  const responses = [
    { lines: numberedLines(0, 1025), next: 1025, running: true },
    { lines: ["final"], next: 1026, running: false },
    { lines: ["new job"], next: 1, running: false },
  ];
  const { job, box, tick } = createHarness(async (url) => {
    urls.push(url);
    return { ok: true, json: async () => responses.shift() };
  });
  const seen = [];
  let complete;
  const done = new Promise((resolve) => { complete = resolve; });
  job.startPolling((data) => {
    seen.push(data.next);
    if (!data.running) complete();
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(box.value, numberedLines(513, 512).join("\n"));
  tick();
  await done;
  assert.deepEqual(urls, ["status?since=0", "status?since=1025"]);
  assert.deepEqual(seen, [1025, 1026]);
  assert.equal(box.value.split("\n").length, 513);
  assert.ok(box.value.endsWith("\nfinal"));
  job.reset();
  assert.equal(box.value, "");
  await new Promise((resolve) => job.startPolling(resolve));
  assert.equal(urls[2], "status?since=0");
  assert.equal(box.value, "new job");
});
