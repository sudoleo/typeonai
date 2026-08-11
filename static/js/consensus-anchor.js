// Pure text/DOM anchor location used by Consensus markers.
(function () {
  window.App = window.App || {};

  function create(stripMarkdown) {
    function normalizeForSearch(value) {
      return String(value || "")
        .toLowerCase()
        .replace(/[“”„‘’«»"]/g, '"')
        .replace(/\s+/g, " ")
        .trim();
    }
    
    // Sucht den normalisierten Needle in einem Rohtext und liefert die
    // Original-Offsets (für splitText/Range) zurück.
    function findRangesInText(raw, normNeedle) {
      let norm = "";
      const map = [];
      for (let i = 0; i < raw.length; i++) {
        let ch = raw[i].toLowerCase();
        if (/[“”„‘’«»"]/.test(ch)) ch = '"';
        if (/\s/.test(ch)) {
          if (norm.endsWith(" ") || norm === "") continue;
          ch = " ";
        }
        norm += ch;
        map.push(i);
      }
      if (!normNeedle) return [];
      const ranges = [];
      let from = 0;
      let idx;
      while ((idx = norm.indexOf(normNeedle, from)) !== -1) {
        ranges.push({ start: map[idx], end: map[idx + normNeedle.length - 1] + 1 });
        from = idx + Math.max(1, normNeedle.length);
      }
      return ranges;
    }
    
    function findRangeInText(raw, normNeedle) {
      return findRangesInText(raw, normNeedle)[0] || null;
    }
    
    function findRangeInTextNode(node, normNeedle) {
      return findRangeInText(node.nodeValue || "", normNeedle);
    }
    
    // Quellentags im Ankertext. Im Konsens-MARKDOWN steht "…330 m.[S1]",
    // im gerenderten DOM ist daraus ein .src-ref-Chip mit dem Text "1"
    // geworden — und der wird beim Markieren uebersprungen. Ein Anker mit
    // Tag fand deshalb nie seine Stelle, obwohl der Consensus-Prompt die
    // Tags ausgerechnet an die ZENTRALEN Faktenaussagen haengt. Eine
    // zusaetzliche tagfreie Variante deckt genau diese Faelle ab.
    // Dieselbe Tag-Form, die linkifySourceTags erkennt ([S1], [1], [S1, S2]).
    const SOURCE_TAG_RE = /\[S?\d+(?:\s*,\s*S?\d+)*\]/gi;
    
    function withoutSourceTags(value) {
      const text = String(value || "");
      if (!SOURCE_TAG_RE.test(text)) {
        SOURCE_TAG_RE.lastIndex = 0;
        return "";
      }
      SOURCE_TAG_RE.lastIndex = 0;
      // Bei alten Bookmarks ("Aussage [S1].") zieht der Renderer das
      // Satzzeichen VOR den Chip. Der Leerraum, der dabei verschwindet,
      // faellt hier ebenfalls weg, sonst passt der Anker erneut nicht.
      return text.replace(SOURCE_TAG_RE, " ").replace(/\s+([.,;:!?])/g, "$1");
    }
    
    // Erst die vollen Varianten (ohne und mit Auszeichnung), dann die
    // gekuerzten. Eine auf acht Woerter gestutzte Variante darf nie vor
    // einem vollstaendigen Treffer gewinnen.
    function searchVariants(text) {
      const full = [];
      const short = [];
      const seen = new Set();
      const stripped = stripMarkdown(text);
      // Die tagfreien Varianten stehen bewusst HINTER den originalen:
      // ein exakter Treffer darf nie gegen eine bereinigte Fassung
      // verlieren.
      const candidates = [stripped, text];
      [stripped, text].forEach(function (candidate) {
        const cleaned = withoutSourceTags(candidate);
        if (cleaned) candidates.push(cleaned);
      });
      candidates.forEach(function (candidate) {
        const norm = normalizeForSearch(candidate)
          .replace(/^(\.{3}|…)\s*/, "").replace(/\s*(\.{3}|…)$/, "");
        if (!norm || seen.has(norm)) return;
        seen.add(norm);
        full.push(norm);
        const words = norm.split(" ");
        if (words.length > 8) {
          const head = words.slice(0, 8).join(" ");
          if (seen.has(head)) return;
          seen.add(head);
          short.push(head);
        }
      });
      return full.concat(short);
    }
    
    // Textknoten, die beim Markieren und bei der Ankersuche uebersprungen
    // werden: Badges/Marker und die [S1]-Quellenchips sind UI (ihr Text
    // wuerde die Offsets verschieben), Code und KaTeX duerfen nicht
    // angefasst werden.
    // `.src-ref` sind die hochgestellten Quellenzahlen im Konsens. Ohne
    // sie hier wurde die Zahl selbst als Satzteil gewrappt und trug dann
    // die Unterstreichung der Passage — eine bernsteinfarbene "3" sieht
    // aus wie ein Fehler, nicht wie eine Fussnote.
    const MARK_SKIP_SELECTOR =
      ".claim-badge, .cx-marker, .source-link, .src-ref, .src-ref-sep, code, pre, .katex";
    const BLOCK_SELECTOR = "p, li, td, th, h1, h2, h3, h4, h5, h6, blockquote, dd, dt";
    
    // --- Inline-Marker: Satzgrenzen und Text-Wrapping -----------------
    // Der verifizierte Anker ist absichtlich kurz (5-12 Wörter). Als
    // Markierungseinheit wäre er zu klein: ein unterstrichenes Fragment
    // mitten im Satz wirkt zufällig. Für die Darstellung wird deshalb auf
    // den umgebenden Satz ausgedehnt, der Anker selbst bleibt intern
    // unverändert (Popover zitiert weiter den Anker).
    
    // Über diese Länge hinaus ist "der ganze Satz" keine Hilfe mehr,
    // sondern eine Wand aus Unterstreichung: dann nur der Anker.
    const MAX_MARK_CHARS = 400;
    
    // Abkürzungen, deren Punkt kein Satzende ist.
    const ABBREVIATIONS = [
      "z.b", "u.a", "d.h", "u.u", "i.d.r", "ggf", "bzw", "ca", "etc", "usw",
      "vgl", "evtl", "inkl", "exkl", "nr", "abb", "tab", "bspw", "dr", "prof",
      "mr", "mrs", "ms", "st", "vs", "approx", "e.g", "i.e", "cf", "fig",
      "no", "inc", "ltd", "co", "al", "jr", "sr", "ph.d"
    ];
    
    function isAbbreviationBefore(text, dotIndex) {
      const before = text.slice(Math.max(0, dotIndex - 12), dotIndex).toLowerCase();
      const word = (before.match(/[a-zäöüß.]+$/) || [""])[0];
      return ABBREVIATIONS.includes(word);
    }
    
    // Ist text[i] ("." / "!" / "?") ein echtes Satzende?
    function isSentenceEnd(text, i) {
      const ch = text[i];
      if (ch === ".") {
        const prev = text[i - 1] || "";
        const next = text[i + 1] || "";
        if (/\d/.test(prev) && /\d/.test(next)) return false;          // 1.5
        // Bei "..." sind die ersten beiden Punkte kein Ende; der letzte
        // wird wie das einzelne Unicode-Ellipsis behandelt.
        if (next === ".") return false;
        // Einzelner Großbuchstabe davor = Initial ("J. R. R.")
        if (/[A-ZÄÖÜ]/.test(prev) && !/[A-Za-zÄÖÜäöüß]/.test(text[i - 2] || " ")) return false;
        if (isAbbreviationBefore(text, i)) return false;
      }
      // Schließende Anführungszeichen/Klammern gehören noch zum Satz.
      let j = i + 1;
      while (j < text.length && /["'”’»)\]]/.test(text[j])) j++;
      if (j >= text.length) return true;
      if (!/\s/.test(text[j])) return false;
      let k = j;
      while (k < text.length && /\s/.test(text[k])) k++;
      if (k >= text.length) return true;
      // Kleinbuchstabe danach spricht gegen einen Satzanfang.
      return !/[a-zäöüß]/.test(text[k]);
    }
    
    // Endet [.., end) bereits auf einem Satzende? Seit der Anker die
    // Satznummer aufloest (statt eines 5-12-Wort-Ausschnitts) ist das der
    // Normalfall — und dann darf nicht weitergedehnt werden, sonst
    // verschluckt der erste Satz den zweiten und beide Claims landen auf
    // derselben Markierung.
    function endsAtSentenceBoundary(text, end) {
      let i = end - 1;
      while (i >= 0 && /["'”’»)\]]/.test(text[i])) i--;
      return i >= 0 && /[.!?…]/.test(text[i]) && isSentenceEnd(text, i);
    }
    
    // Dehnt [start,end) auf die umgebenden Satzgrenzen aus.
    function sentenceBounds(text, start, end) {
      let s = start;
      while (s > 0) {
        const i = s - 1;
        if (text[i] === "\n") break;
        if (/[.!?…]/.test(text[i]) && isSentenceEnd(text, i)) break;
        s--;
      }
      while (s < end && /\s/.test(text[s])) s++;
    
      let e = end;
      while (!endsAtSentenceBoundary(text, e) && e < text.length) {
        const ch = text[e];
        if (ch === "\n") break;
        if (/[.!?…]/.test(ch) && isSentenceEnd(text, e)) {
          e++;
          while (e < text.length && /["'”’»)\]]/.test(text[e])) e++;
          break;
        }
        e++;
      }
      while (e > s && /\s/.test(text[e - 1])) e--;
    
      if (e - s > MAX_MARK_CHARS) return { start: start, end: end };
      return { start: s, end: e };
    }
    
    // Blockelement, in dem ein Textknoten steht (Grenze der Ausdehnung).
    function blockOf(node, container) {
      const el = node.parentElement;
      if (!el) return container;
      const block = el.closest(BLOCK_SELECTOR);
      return (block && container.contains(block)) ? block : container;
    }
    
    // Flache Textsicht eines Blocks: alle markierbaren Textknoten in
    // Dokumentreihenfolge mit ihren Offsets im zusammengesetzten Text.
    function collectTextSlices(block) {
      const walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT);
      const slices = [];
      let flat = "";
      let node;
      while ((node = walker.nextNode())) {
        const raw = node.nodeValue || "";
        if (!raw) continue;
        if (node.parentElement?.closest(MARK_SKIP_SELECTOR)) continue;
        slices.push({ node, start: flat.length, end: flat.length + raw.length });
        flat += raw;
      }
      return { slices, flat };
    }
    
    // Anker -> konkreter Bereich in der flachen Textsicht seines Blocks.
    function locateAnchor(container, anchor, occurrence) {
      const wanted = Number.isInteger(occurrence) && occurrence >= 0 ? occurrence : 0;
      const blocks = [];
      const seenBlocks = new Set();
      const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
      let node;
      while ((node = walker.nextNode())) {
        if (!node.nodeValue || node.parentElement?.closest(MARK_SKIP_SELECTOR)) continue;
        const block = blockOf(node, container);
        if (!seenBlocks.has(block)) {
          seenBlocks.add(block);
          blocks.push(block);
        }
      }
    
      for (const needle of searchVariants(anchor)) {
        const hits = [];
        blocks.forEach(function (block) {
          const view = collectTextSlices(block);
          if (!view.slices.length) return;
          findRangesInText(view.flat, needle).forEach(function (range) {
            hits.push({
              block: block,
              slices: view.slices,
              flat: view.flat,
              start: range.start,
              end: range.end
            });
          });
        });
        if (hits[wanted]) return hits[wanted];
        if (occurrence == null && hits[0]) return hits[0];
      }
      return null;
    }
    
    // Wrappt [start,end) der flachen Sicht in <span class="...">.
    // Bewusst pro Textknoten statt per Range.extractContents: erhaltene
    // Inline-Auszeichnung (<strong>, [S1]-Links, KaTeX) bleibt exakt an
    // ihrem Platz, es wird nichts umgehängt.
    return Object.freeze({
      normalizeForSearch,
      findRangesInText,
      findRangeInText,
      findRangeInTextNode,
      searchVariants,
      sentenceBounds,
      locateAnchor
    });
  }

  window.App.consensusAnchor = Object.freeze({ create });
})();

