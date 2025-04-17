import cv2
import os
import time
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

def load_images_from_sequence(folder, prefix):
    images = []
    filenames = sorted([f for f in os.listdir(folder) if f.startswith(prefix)])
    for f in filenames:
        img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append((f, img))
    return images

def compute_descriptors(images):
    sift = cv2.SIFT_create()
    descriptors = []
    for name, img in images:
        kp, des = sift.detectAndCompute(img, None)
        descriptors.append((name, des))
    return descriptors

def match_images(descriptorsA, descriptorsB):
    bf = cv2.BFMatcher() # Brute-force matcher pour comparer les descripteurs
    matches = []
    total_time = 0

    for i, (nameA, desA) in enumerate(descriptorsA):
        best_match_idx = -1
        best_match_score = float('inf')

        start = time.time()
        for j, (nameB, desB) in enumerate(descriptorsB):
            # Si un des deux descripteurs est vide, on l'ignore
            if desA is None or desB is None:
                continue

            # Trouver les deux meilleurs matchs pour chaque descripteur
            m = bf.knnMatch(desA, desB, k=2)

            # Test du ratio de Lowe pour garder les bons matchs
            good = [m1 for m1, m2 in m if m1.distance < 0.75 * m2.distance]

            # Le score est l'opposé du nombre de bons matchs
            # (on veut maximiser le nombre de bons matchs, donc minimiser le score)
            score = -len(good)

            # Si ce score est meilleur (plus négatif = plus de bons matchs), on le garde
            if score < best_match_score:
                best_match_score = score
                best_match_idx = j

        end = time.time()
        total_time += (end - start)

        matches.append((i, best_match_idx))

    avg_time = total_time / len(descriptorsA)
    print(f"Temps moyen de matching : {avg_time:.4f} s")

    return matches

def compute_error(matches):
    errors = [abs(i - j) for i, j in matches if j != -1]
    return np.mean(errors), errors

# Afficher les correspondances entre deux images
def draw_matches(imgA, kpA, imgB, kpB, good_matches, out_path):
    match_img = cv2.drawMatches(
        imgA, kpA, imgB, kpB, good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(out_path, match_img)

# Compter les keypoints et les "good matches"
def analyze_keypoints(imagesA, imagesB):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    stats = []

    for (nameA, imgA), (nameB, imgB) in zip(imagesA, imagesB):
        kpA, desA = sift.detectAndCompute(imgA, None)
        kpB, desB = sift.detectAndCompute(imgB, None)

        if desA is None or desB is None:
            continue

        matches = bf.knnMatch(desA, desB, k=2)
        good = [m1 for m1, m2 in matches if m1.distance < 0.75 * m2.distance]

        stats.append({
            "imgA": nameA,
            "imgB": nameB,
            "kpA": len(kpA),
            "kpB": len(kpB),
            "good_matches": len(good)
        })

    df = pd.DataFrame(stats)
    df.to_csv(f"results/sift/BFMatcher/{seq_name}/sift_keypoint_stats_{seq_name}.csv", index=False)
    print(df.head())

if __name__ == "__main__":
    path = "data/"
    seq_name = "visages"  # magasin, brain, studio, neige, parc ou legumes

    sys.stdout = open(f"results/sift/BFMatcher/{seq_name}/log_{seq_name}.txt", "w")

    sessionA = load_images_from_sequence(path, f"{seq_name}A")
    sessionB = load_images_from_sequence(path, f"{seq_name}B")

    descriptorsA = compute_descriptors(sessionA)
    descriptorsB = compute_descriptors(sessionB)

    # A -> B
    print("A -> B")
    matches = match_images(descriptorsA, descriptorsB)
    mean_error, all_errors = compute_error(matches)

    print(f"Erreur moyenne : {mean_error}")
    for (i, j), err in zip(matches, all_errors):
        print(f"A{i} <--> B{j} | Erreur = {err}")

    # B -> A
    print("\nB -> A")
    reverse_matches = match_images(descriptorsB, descriptorsA)
    reverse_mean_error, reverse_errors = compute_error(reverse_matches)

    print(f"Erreur moyenne B->A : {reverse_mean_error}")
    for (i, j), err in zip(reverse_matches, reverse_errors):
        print(f"B{i} <--> A{j} | Erreur = {err}")

    total_error = mean_error + reverse_mean_error
    print(f"\nErreur totale (A->B + B->A) : {total_error}")

    # Exemple : visualiser les matches entre A3 et B3
    idx = 3
    sift = cv2.SIFT_create()
    imgA = sessionA[idx][1]
    imgB = sessionB[idx][1]
    kpA, desA = sift.detectAndCompute(imgA, None)
    kpB, desB = sift.detectAndCompute(imgB, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desA, desB, k=2)
    good = [m1 for m1, m2 in matches if m1.distance < 0.75 * m2.distance]

    draw_matches(imgA, kpA, imgB, kpB, good, f"results/sift/BFMatcher/{seq_name}/matches_{seq_name}_A{idx}_B{idx}.png")

# Graphiques
plt.figure(figsize=(10, 4))
plt.plot(all_errors, label="A->B")
plt.plot(reverse_errors, label="B->A")
plt.xlabel("Index image")
plt.ylabel("Erreur de position")
plt.title("Erreur de localisation par SIFT ("+seq_name+")")
plt.legend()
plt.tight_layout()
plt.savefig("results/sift/BFMatcher/"+seq_name+"/erreurs_sift_"+seq_name+".png")
plt.show()

# Sauvegarder les résultats dans des CSV
df = pd.DataFrame(matches, columns=["A_index", "B_index"])
df["erreur"] = df.apply(lambda row: abs(row["A_index"].trainIdx - row["B_index"].trainIdx), axis=1)
df.to_csv(f"results/sift/BFMatcher/{seq_name}/sift_matches_{seq_name}_A_to_B.csv", index=False)

df2 = pd.DataFrame(reverse_matches, columns=["B_index", "A_index"])
df2["erreur"] = df2.apply(lambda row: abs(row["B_index"] - row["A_index"]), axis=1)
df2.to_csv(f"results/sift/BFMatcher/{seq_name}/sift_matches_{seq_name}_B_to_A.csv", index=False)

# Sauvegarder le nombre de keypoints et de bons matchs dans un CSV
analyze_keypoints(sessionA, sessionB)

sys.stdout.close()
sys.stdout = sys.__stdout__
print(f"Résultats sauvegardés dans : results/sift/BFMatcher/{seq_name}/log_{seq_name}.txt")