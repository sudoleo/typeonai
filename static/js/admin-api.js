export function createAdminClient(auth) {
  return async function adminRequest(method, path, body) {
    const user = auth.currentUser;
    if (!user) throw new Error("Not logged in");
    const idToken = await user.getIdToken();
    const response = await fetch(path, {
      method,
      headers: {
        "Authorization": `Bearer ${idToken}`,
        "Content-Type": "application/json"
      },
      body: body ? JSON.stringify(body) : undefined
    });
    let data = {};
    try { data = await response.json(); } catch (_) { /* empty */ }
    if (!response.ok) {
      // FastAPI-Fehler koennen ein Objekt sein ({error_code, message} oder
      // eine Validierungsliste). Ohne Auspacken stuende hier "[object Object]".
      const detail = data.detail;
      const message = data.error
        || (typeof detail === "string" ? detail : null)
        || (detail && typeof detail === "object" && !Array.isArray(detail)
          ? (detail.message || detail.error)
          : null)
        || `HTTP ${response.status}`;
      throw new Error(message);
    }
    return data;
  };
}

