/* Topic page: the check strip readout and the returning-reader band.
 *
 * Both are progressive: the page is complete without this file. The strip is
 * rendered server-side and stays readable as a shape; the band is empty until
 * this browser proves it has been here before. Nothing is sent anywhere -- the
 * last-visit mark lives in this browser only.
 */
(function () {
  "use strict";

  var STORE_PREFIX = "topic-seen:";

  function readStrip() {
    var strip = document.getElementById("topicStrip");
    var read = document.getElementById("topicStripRead");
    if (!strip || !read) return null;
    var cells = Array.prototype.slice.call(
      strip.querySelectorAll(".topic-strip-cell")
    );
    var resting = read.innerHTML;

    // Without a pointer that can hover there is no way to read a cell before
    // following it, so the first tap previews and the second one opens.
    var canHover = !window.matchMedia || window.matchMedia("(hover: hover)").matches;
    var previewed = null;

    function show(cell) {
      var parts = ["<b>" + cell.dataset.date + "</b> &mdash; " + cell.dataset.note];
      if (cell.dataset.score) {
        parts.push('<span class="topic-strip-score">' + cell.dataset.score + "/100 agreement</span>");
      }
      read.innerHTML = parts.join(" ");
    }

    cells.forEach(function (cell) {
      cell.addEventListener("mouseenter", function () { show(cell); });
      cell.addEventListener("focus", function () { show(cell); });
      cell.addEventListener("click", function (event) {
        if (!canHover && previewed !== cell) {
          event.preventDefault();
          previewed = cell;
          show(cell);
        }
      });
    });
    strip.addEventListener("mouseleave", function () { read.innerHTML = resting; });
    return {strip: strip, cells: cells};
  }

  function store(key, value) {
    try {
      window.localStorage.setItem(key, value);
    } catch (error) {
      /* Private mode or blocked storage: the band simply never appears. */
    }
  }

  function readStore(key) {
    try {
      return window.localStorage.getItem(key);
    } catch (error) {
      return null;
    }
  }

  function returningReader(strip, cells) {
    var band = document.getElementById("topicReturn");
    if (!band || !cells.length) return;
    // Browsing an older version must not mark the newer checks as seen.
    if (window.location.search.indexOf("version=") !== -1) return;
    var key = STORE_PREFIX + (strip.dataset.slug || "");
    var seen = readStore(key);
    var latest = cells[cells.length - 1].dataset.iso || "";

    if (seen && latest && seen < latest) {
      var fresh = cells.filter(function (cell) {
        return (cell.dataset.iso || "") > seen;
      });
      if (fresh.length) {
        fresh.forEach(function (cell) { cell.classList.add("is-unseen"); });
        var moved = fresh.filter(function (cell) {
          return cell.dataset.kind === "material";
        });
        var stirred = fresh.filter(function (cell) {
          return cell.dataset.kind === "event";
        });
        var count = fresh.length + " check" + (fresh.length === 1 ? "" : "s");
        var line;
        if (moved.length) {
          band.classList.add("is-moved");
          line = count + " since your last visit, and the answer moved on " +
            moved[moved.length - 1].dataset.date + ".";
        } else if (stirred.length) {
          line = count + " since your last visit. The answer held; " +
            stirred.length + " of those check" + (stirred.length === 1 ? "" : "s") +
            " changed which statements were made.";
        } else {
          line = count + " since your last visit. The answer did not move.";
        }
        band.innerHTML =
          '<span class="topic-return-tag">Since your last visit</span>' +
          "<p>" + line + ' <a href="#facts">See the statements</a></p>';
        band.hidden = false;
      }
    }
    if (latest) store(key, latest);
  }

  function start() {
    var found = readStrip();
    if (!found) return;
    returningReader(found.strip, found.cells);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
