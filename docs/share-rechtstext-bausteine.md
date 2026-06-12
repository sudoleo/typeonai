# Rechtstext-Bausteine: Öffentliches Teilen (Share-Feature)

Entwurfs-Bausteine für Terms und Privacy. **Nicht automatisch eingebaut** —
zur manuellen Übernahme in `templates/terms.html` und `templates/privacy.html`.
Stand: 2026-06-12 (Etappe 3 des Share-Features).

---

## Baustein 1 — Terms: Öffentliches Teilen von Konsens-Antworten

> **Public sharing of consensus answers**
>
> Logged-in users can publish a consensus answer as a public, read-only page
> ("shared page"). Creating a shared page is always an explicit, opt-in action;
> nothing is published automatically.
>
> A shared page contains the question, the consensus answer, the differences
> analysis, the list of sources, and the names of the AI models consulted —
> as a snapshot at the time of sharing. It does not contain your name, e-mail
> address, or any other account data, and it is not publicly linked to your
> account.
>
> By creating a shared page you confirm that the question and any content you
> contributed to it do not infringe third-party rights (including copyright
> and privacy rights) and are not unlawful. You can revoke any of your shared
> pages at any time under "My shared links"; revoked pages immediately stop
> being publicly available.
>
> We may remove (block) shared pages that violate these Terms or applicable
> law, and we may limit the number of pages a user can create per day.
> Search-engine indexing of shared pages is disabled by default and is only
> enabled for selected pages after a manual review by us.

## Baustein 2 — Terms: Melden von Inhalten (Moderation)

> **Reporting shared pages**
>
> Every shared page offers a "Report this page" function that can be used
> without an account. Reports are stored only as an anonymous counter per
> reason (e.g. "inaccurate", "harmful", "spam"); we do not store the
> reporter's IP address, browser data, or any other personal data with a
> report. Repeated reports automatically remove a page from search-engine
> indexing until we have reviewed it. There is no obligation to act on every
> report, but we review reported pages with priority.

## Baustein 3 — Privacy: Datenverarbeitung beim Teilen

> **Shared pages (public sharing)**
>
> When you share a consensus answer, we store the shared content (question,
> consensus answer, differences, sources, model names) together with your
> internal account ID. The account ID is used solely so that you can manage
> and revoke your own shared pages and so that we can fulfil our moderation
> duties; it is never displayed publicly and never embedded in the public page.
>
> Shared pages are public: anyone with the link can read them, and selected
> pages may appear in search engines after manual review. Please do not include
> personal data in questions you intend to share.

## Baustein 4 — Privacy: Speicher- und Löschfristen

> **Retention and deletion**
>
> - Consensus results that are eligible for sharing are kept server-side for
>   a maximum of **24 hours**; if you do not share them within that period,
>   they are deleted automatically.
> - Shared pages remain online until you revoke them or delete your account.
> - When you **revoke** a shared page, it immediately becomes unavailable to
>   the public (HTTP 410) and is **permanently deleted within 30 days**.
>   Note: copies cached by browsers or search engines for a short period are
>   outside our control.
> - When you **delete your account**, all your shared pages and pending
>   results are deleted as part of the account-deletion process.
> - Reports on shared pages are stored without any personal data (anonymous
>   counters only) and are deleted together with the page.

## Baustein 5 — Privacy (optional, Kurzfassung für die Übersichtstabelle)

> | Data | Purpose | Retention |
> | --- | --- | --- |
> | Shared page content (question, answers, sources, model names) | Public sharing, chosen by you | Until revoked (then ≤ 30 days) or account deletion |
> | Pending consensus results | Enable the "Share" button after a run | Max. 24 hours |
> | Report counters per shared page | Moderation | Lifetime of the page, anonymous |

---

### Hinweise zur Übernahme (nicht veröffentlichen)

- Baustein 1 + 2 gehören in die Terms (Abschnitt Nutzungsregeln / Inhalte).
- Baustein 3 + 4 gehören in die Privacy Policy; Baustein 5 nur, falls die
  Privacy-Seite eine Übersichtstabelle hat.
- Die 30-Tage-Frist ist im Code als `REVOKED_RETENTION_DAYS` in
  `app/services/share_snapshots.py` hinterlegt; die 24h als `PENDING_TTL_HOURS`.
  Bei Textänderungen an den Fristen beide Stellen konsistent halten.
- Kein IP-/UA-Logging bei Reports ist im Code zugesichert
  (`report_share()` speichert nur Zähler + Grund) — Formulierungen in
  Baustein 2/4 dürfen das daher fest zusagen.
