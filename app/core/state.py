# Der einzige verbleibende In-Memory-State ist der kurzlebige Feedback-
# Cooldown. Produktive Usage liegt ausschliesslich in Firestore
# (app/services/usage_repository.py).
last_feedback_time = {}  # {uid: timestamp}
