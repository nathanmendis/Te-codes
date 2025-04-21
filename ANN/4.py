import numpy as np
print("Natha Mendis TE-AIML 33543")
print("script started")
class ART1:
    def __init__(self, input_size, vigilance=0.7):
        self.vigilance = vigilance
        self.weights = []
        self.input_size = input_size

    def train(self, X):
        for idx, x in enumerate(X):
            x = np.array(x)
            match_found = False
            print(f"\n🧩 Input Pattern {idx+1}: {x}")
            for i, w in enumerate(self.weights):
                intersection = np.minimum(x, w)
                match_score = np.sum(intersection) / (np.sum(x) + 1e-5)
                print(f"  🔍 Comparing with Cluster {i+1}: {w}")
                print(f"     ∩ Intersection: {intersection}")
                print(f"     ✅ Match Score: {match_score:.4f}")
                if match_score >= self.vigilance:
                    print(f"     🔗 Matched! Updating Cluster {i+1} to intersection.")
                    self.weights[i] = intersection
                    match_found = True
                    break
                else:
                    print(f"     ❌ No Match (score < {self.vigilance})")
            if not match_found:
                print(f"  ➕ No match found. Creating new cluster.")
                self.weights.append(x)

        print("\n📦 Final Clusters:")
        for i, w in enumerate(self.weights):
            print(f"  Cluster {i+1}: {w}")

# Sample binary input patterns
X = np.array([
    [1, 0, 1, 0],
    [1, 1, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1]
])

art = ART1(input_size=4, vigilance=0.7)
art.train(X)
