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
      throw new Error(data.error || data.detail || `HTTP ${response.status}`);
    }
    return data;
  };
}

