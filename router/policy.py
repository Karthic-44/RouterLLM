import numpy as np


def cost_order(costs):
    return sorted(costs, key=lambda a: costs[a])


def prob_true(model, X):
    classes = list(model.classes_)
    if len(classes) == 1:
        return np.full(len(X), float(classes[0]))
    return model.predict_proba(X)[:, classes.index(1)]


def choose(probs, order, threshold):
    for a in order:
        if probs[a] >= threshold:
            return a
    return max(order, key=lambda a: probs[a])


def choose_many(P, actions, order, threshold):
    col = {a: i for i, a in enumerate(actions)}
    pick = np.full(len(P), -1)
    for a in order:
        free = (pick == -1) & (P[:, col[a]] >= threshold)
        pick[free] = col[a]
    left = pick == -1
    if left.any():
        cols = np.array([col[a] for a in order])
        best = cols[P[:, cols].argmax(axis=1)]
        pick[left] = best[left]
    return pick
