history = []

embedding = model.encode(text)        # BERT embedding
embedding = torch.tensor(embedding)

h = to_poincare(embedding)

# apply focus field BEFORE scoring
h = focus_force(h, history, strength=0.03)
h = renormalize_poincare(h)

score, diag = siphon_score(h, history)

history.append(h.detach())

print(diag)
