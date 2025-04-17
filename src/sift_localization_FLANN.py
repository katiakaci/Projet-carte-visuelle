import cv2
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

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
        descriptors.append((name, kp, des))
    return descriptors

def match_images(descriptorsA, descriptorsB, flann):
    matches = []
    total_time = 0

    for i, (nameA, kpA, desA) in enumerate(descriptorsA):
        best_match_idx = -1
        best_match_score = float('inf')

        start = time.time()
        for j, (nameB, kpB, desB) in enumerate(descriptorsB):
            if desA is None or desB is None:
                continue

            if desA.dtype != np.float32:
                desA = desA.astype(np.float32)
            if desB.dtype != np.float32:
                desB = desB.astype(np.float32)

            m = flann.knnMatch(desA, desB, k=2)
            good = [m1 for m1, m2 in m if m1.distance < 0.75 * m2.distance]
            score = -len(good)

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

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    for seq_name in sequences:
        print(f"\nTraitement de la séquence : {seq_name}")
        os.makedirs(f"results/sift/FLANN/{seq_name}", exist_ok=True)
        sys.stdout = open(f"results/sift/FLANN/{seq_name}/log_{seq_name}.txt", "w")

        sessionA = load_images_from_sequence(path, f"{seq_name}A")
        sessionB = load_images_from_sequence(path, f"{seq_name}B")

        descriptorsA = compute_descriptors(sessionA)
        descriptorsB = compute_descriptors(sessionB)

        print("A -> B")
        matches = match_images(descriptorsA, descriptorsB, flann)
        mean_error, all_errors = compute_error(matches)
        print(f"Erreur moyenne : {mean_error}")
        for (i, j), err in zip(matches, all_errors):
            print(f"A{i} <--> B{j} | Erreur = {err}")

        print("\nB -> A")
        reverse_matches = match_images(descriptorsB, descriptorsA, flann)
        reverse_mean_error, reverse_errors = compute_error(reverse_matches)
        print(f"Erreur moyenne B->A : {reverse_mean_error}")
        for (i, j), err in zip(reverse_matches, reverse_errors):
            print(f"B{i} <--> A{j} | Erreur = {err}")

        total_error = mean_error + reverse_mean_error
        print(f"\nErreur totale (A->B + B->A) : {total_error}")

        # Graphique des erreurs
        plt.figure(figsize=(10, 4))
        plt.plot(all_errors, label="A->B")
        plt.plot(reverse_errors, label="B->A")
        plt.xlabel("Index image")
        plt.ylabel("Erreur de position")
        plt.title("Erreur de localisation par SIFT + FLANN ("+seq_name+")")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/sift/FLANN/{seq_name}/erreurs_sift_{seq_name}.png")
        plt.close()

        # Histogramme
        plt.hist(all_errors + reverse_errors, bins=range(0, max(all_errors + reverse_errors)+2), edgecolor='black')
        plt.title(f"Histogramme des erreurs (total A→B et B→A) – {seq_name}")
        plt.xlabel("Erreur |i - j|")
        plt.ylabel("Fréquence")
        plt.tight_layout()
        plt.savefig(f"results/sift/FLANN/{seq_name}/histogramme_erreurs_{seq_name}.png")
        plt.close()

        # Matrice de confusion
        confusion = np.zeros((len(sessionA), len(sessionB)))
        for i, j in matches:
            if j != -1:
                confusion[i, j] += 1
        plt.imshow(confusion, cmap='viridis', interpolation='nearest')
        plt.title(f"Matrice de confusion A→B – {seq_name}")
        plt.xlabel("Index B prédit")
        plt.ylabel("Index A")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"results/sift/FLANN/{seq_name}/confusion_matrix_{seq_name}_A_to_B.png")
        plt.close()

        # Sauvegarde CSV
        df = pd.DataFrame(matches, columns=["A_index", "B_index"])
        df["erreur"] = df.apply(lambda row: abs(row["A_index"] - row["B_index"]), axis=1)
        df.to_csv(f"results/sift/FLANN/{seq_name}/sift_matches_{seq_name}_A_to_B.csv", index=False)

        df2 = pd.DataFrame(reverse_matches, columns=["B_index", "A_index"])
        df2["erreur"] = df2.apply(lambda row: abs(row["B_index"] - row["A_index"]), axis=1)
        df2.to_csv(f"results/sift/FLANN/{seq_name}/sift_matches_{seq_name}_B_to_A.csv", index=False)

        # Visualisation A3-B3
        idx = 3
        sift = cv2.SIFT_create()
        imgA = sessionA[idx][1]
        imgB = sessionB[idx][1]
        kpA, desA = sift.detectAndCompute(imgA, None)
        kpB, desB = sift.detectAndCompute(imgB, None)
        if desA is not None and desB is not None:
            if desA.dtype != np.float32:
                desA = desA.astype(np.float32)
            if desB.dtype != np.float32:
                desB = desB.astype(np.float32)
            m = flann.knnMatch(desA, desB, k=2)
            good = [m1 for m1, m2 in m if m1.distance < 0.75 * m2.distance]
            match_img = cv2.drawMatches(imgA, kpA, imgB, kpB, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite(f"results/sift/FLANN/{seq_name}/matches_{seq_name}_A{idx}_B{idx}.png", match_img)

        # Statistiques keypoints
        stats = []
        for (nameA, imgA), (nameB, imgB) in zip(sessionA, sessionB):
            kpA, desA = sift.detectAndCompute(imgA, None)
            kpB, desB = sift.detectAndCompute(imgB, None)
            if desA is None or desB is None:
                continue
            if desA.dtype != np.float32:
                desA = desA.astype(np.float32)
            if desB.dtype != np.float32:
                desB = desB.astype(np.float32)
            matches_kp = flann.knnMatch(desA, desB, k=2)
            good = [m1 for m1, m2 in matches_kp if m1.distance < 0.75 * m2.distance]
            stats.append({
                "imgA": nameA,
                "imgB": nameB,
                "kpA": len(kpA),
                "kpB": len(kpB),
                "good_matches": len(good)
            })
        pd.DataFrame(stats).to_csv(f"results/sift/FLANN/{seq_name}/sift_keypoint_stats_{seq_name}.csv", index=False)

        sys.stdout.close()
        sys.stdout = sys.__stdout__
        print(f"Résultats pour {seq_name} sauvegardés.")
