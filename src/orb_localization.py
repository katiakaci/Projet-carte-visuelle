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
    orb = cv2.ORB_create()
    descriptors = []
    for name, img in images:
        kp, des = orb.detectAndCompute(img, None)
        descriptors.append((name, kp, des))
    return descriptors

def match_images(descriptorsA, descriptorsB):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = []
    total_time = 0

    for i, (nameA, kpA, desA) in enumerate(descriptorsA):
        best_match_idx = -1
        best_score = -1

        start = time.time()
        for j, (nameB, kpB, desB) in enumerate(descriptorsB):
            if desA is None or desB is None:
                continue
            good = bf.match(desA, desB)
            score = len(good)

            if score > best_score:
                best_score = score
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

def draw_matches(imgA, kpA, imgB, kpB, good_matches, out_path):
    match_img = cv2.drawMatches(imgA, kpA, imgB, kpB, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(out_path, match_img)

def analyze_keypoints(imagesA, imagesB, seq_name):
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    stats = []

    for (nameA, imgA), (nameB, imgB) in zip(imagesA, imagesB):
        kpA, desA = orb.detectAndCompute(imgA, None)
        kpB, desB = orb.detectAndCompute(imgB, None)

        if desA is None or desB is None:
            continue

        matches = bf.match(desA, desB)
        stats.append({
            "imgA": nameA,
            "imgB": nameB,
            "kpA": len(kpA),
            "kpB": len(kpB),
            "good_matches": len(matches)
        })

    df = pd.DataFrame(stats)
    df.to_csv(f"results/orb/{seq_name}/orb_keypoint_stats_{seq_name}.csv", index=False)
    print(df.head())

if __name__ == "__main__":
    path = "data/"
    sequences = ["brain", "legumes", "studio", "visages", "parc", "neige", "magasin"]
    os.makedirs("results/orb/", exist_ok=True)

    
    for seq_name in sequences:
        log_path = f"results/orb/{seq_name}/log_{seq_name}.txt"
        sys.stdout = open(log_path, "w")

        print(f"\nTraitement de la séquence : {seq_name}")
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

        print(f"\nErreur totale (A->B + B->A) : {mean_error + reverse_mean_error}")

        
        # Histogramme des erreurs
        combined_errors = all_errors + reverse_errors
        if combined_errors:
            plt.hist(combined_errors, bins=range(0, max(combined_errors)+2), edgecolor='black')
            plt.title(f"Histogramme des erreurs (A→B et B→A) – {seq_name}")
            plt.xlabel("Erreur |i - j|")
            plt.ylabel("Fréquence")
            plt.tight_layout()
            plt.savefig(f"results/orb/{seq_name}/histogramme_erreurs_{seq_name}.png")
            plt.close()

        # Matrice de confusion A->B
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
        plt.savefig(f"results/orb/{seq_name}/confusion_matrix_{seq_name}_A_to_B.png")
        plt.close()
        
        # Visualisation de A3-B3
        idx = 3
        orb = cv2.ORB_create()
        imgA = sessionA[idx][1]
        imgB = sessionB[idx][1]
        kpA, desA = orb.detectAndCompute(imgA, None)
        kpB, desB = orb.detectAndCompute(imgB, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches_visu = bf.match(desA, desB)
        draw_matches(imgA, kpA, imgB, kpB, matches_visu, f"results/orb/{seq_name}/matches_{seq_name}_A{idx}_B{idx}.png")

        analyze_keypoints(sessionA, sessionB, seq_name)

        sys.stdout.close()
        sys.stdout = sys.__stdout__
