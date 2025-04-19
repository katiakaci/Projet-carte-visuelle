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
        kp, des = sift.detectAndCompute(img, None)  # détection des extrema avec DoG
        descriptors.append((name, kp, des))
    return descriptors

def match_images(descriptorsA, descriptorsB):
    bf = cv2.BFMatcher() # Brute-force matcher pour comparer les descripteurs
    matches = []
    total_time = 0

    for i, (nameA, kpA, desA) in enumerate(descriptorsA):
        best_match_idx = -1
        best_match_score = float('inf')

        start = time.time()
        for j, (nameB, kpB, desB) in enumerate(descriptorsB):
            if desA is None or desB is None:
                continue

            # Trouver les 2 meilleurs matchs pour chaque descripteur
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

if __name__ == "__main__":
    path = "data/"
    sequences = ["studio", "visages", "legumes", "parc", "neige", "magasin", "brain"]

    for seq_name in sequences:
        print(f"\nTraitement de la séquence : {seq_name}")
        os.makedirs(f"results/sift/BFMatcher/{seq_name}", exist_ok=True)
        sys.stdout = open(f"results/sift/BFMatcher/{seq_name}/log_{seq_name}.txt", "w")

        sessionA = load_images_from_sequence(path, f"{seq_name}A")
        sessionB = load_images_from_sequence(path, f"{seq_name}B")

        descriptorsA = compute_descriptors(sessionA)
        descriptorsB = compute_descriptors(sessionB)

        print("A -> B")
        matches = match_images(descriptorsA, descriptorsB)
        mean_error, all_errors = compute_error(matches)
        print(f"Erreur moyenne : {mean_error}")
        for (i, j), err in zip(matches, all_errors):
            print(f"A{i} <--> B{j} | Erreur = {err}")

        print("\nB -> A")
        reverse_matches = match_images(descriptorsB, descriptorsA)
        reverse_mean_error, reverse_errors = compute_error(reverse_matches)
        print(f"Erreur moyenne B->A : {reverse_mean_error}")
        for (i, j), err in zip(reverse_matches, reverse_errors):
            print(f"B{i} <--> A{j} | Erreur = {err}")

        total_error = mean_error + reverse_mean_error
        print(f"\nErreur totale (A->B + B->A) : {total_error}")

        # Statistiques keypoints (affiché dans le log + CSV)
        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher()
        stats = []
        for (nameA, imgA), (nameB, imgB) in zip(sessionA, sessionB):
            kpA, desA = sift.detectAndCompute(imgA, None)
            kpB, desB = sift.detectAndCompute(imgB, None)
            if desA is None or desB is None:
                continue
            matches_kp = bf.knnMatch(desA, desB, k=2)
            good = [m1 for m1, m2 in matches_kp if m1.distance < 0.75 * m2.distance]
            stats.append({
                "imgA": nameA,
                "imgB": nameB,
                "kpA": len(kpA),
                "kpB": len(kpB),
                "good_matches": len(good)
            })
        df_stats = pd.DataFrame(stats)
        print("\n" + df_stats.head().to_string(index=False))
        df_stats.to_csv(f"results/sift/BFMatcher/{seq_name}/sift_keypoint_stats_{seq_name}.csv", index=False)

        sys.stdout.close()
        sys.stdout = sys.__stdout__
        print(f"Résultats pour {seq_name} sauvegardés.")
