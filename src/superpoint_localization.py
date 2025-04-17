import os
import torch
import numpy as np
import cv2
import time
import sys
import pandas as pd
import matplotlib.pyplot as plt
from lightglue import SuperPoint, LightGlue
from lightglue.utils import match_pair

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_images(folder, prefix):
    filenames = sorted([f for f in os.listdir(folder) if f.startswith(prefix)])
    images = []
    for f in filenames:
        path = os.path.join(folder, f)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (640, 480))
            images.append((f, img))
    return images

def extract_features(model, image):
    img_tensor = torch.from_numpy(image / 255.).float()[None, None].to(device)
    with torch.no_grad():
        # feats = model(img_tensor)
        feats = model({"image": img_tensor})
    return feats

def match_features(model, featsA, featsB):
    with torch.no_grad():
        matches = model({"image0": featsA, "image1": featsB})
    return matches

def compute_matches(imagesA, imagesB, superpoint, matcher):
    start_total = time.time()
    matches = []
    stats = []

    featsA = [extract_features(superpoint, img) for _, img in imagesA]
    featsB = [extract_features(superpoint, img) for _, img in imagesB]

    for i, featA in enumerate(featsA):
        best_j = -1
        best_score = -1
        best_match = None

        for j, featB in enumerate(featsB):
            out = match_features(matcher, featA, featB)
            nb_matches = (out["matches0"] > -1).sum().item()
            if nb_matches > best_score:
                best_score = nb_matches
                best_j = j
                best_match = out

        matches.append((i, best_j))
        stats.append({
            "imgA": imagesA[i][0],
            "imgB": imagesB[best_j][0],
            "keypointsA": featA["keypoints"].shape[0],
            "keypointsB": featsB[best_j]["keypoints"].shape[0],
            "good_matches": best_score
        })

    end_total = time.time()
    print(f"Temps moyen pour {len(imagesA)} comparaisons : {(end_total - start_total)/len(imagesA):.4f} s")
    return matches, stats

def compute_error(matches):
    errors = [abs(i - j) for i, j in matches if j != -1]
    return np.mean(errors), errors

def draw_superpoint_matches(img1, img2, feats1, feats2, matches, out_path):
    kpts1 = feats1["keypoints"].cpu().numpy().squeeze(0)
    kpts2 = feats2["keypoints"].cpu().numpy().squeeze(0)
    match_idxs = matches["matches0"].cpu().numpy().squeeze()

    print(f"matches0 shape: {match_idxs.shape}, kpts2 shape: {kpts2.shape}")

    valid = (match_idxs > -1) & (match_idxs < len(kpts2))
    pts1 = kpts1[valid]
    pts2 = kpts2[match_idxs[valid]]

    print(f"Valid matches: {len(pts1)} / {len(match_idxs)}")

    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    h1, w1 = img1.shape
    h2, w2 = img2.shape
    out = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    out[:, :w1] = img1_color
    out[:, w1:] = img2_color

    from matplotlib import colormaps
    colors = colormaps.get_cmap('hsv')
    colors = colors(np.linspace(0, 1, len(pts1)))

    for i, (p1, p2) in enumerate(zip(pts1, pts2)):
        pt1 = tuple(np.round(p1).astype(int))
        pt2 = tuple(np.round(p2).astype(int) + np.array([w1, 0]))
        color = tuple((np.array(colors[i][:3]) * 255).astype(int).tolist())

        cv2.line(out, pt1, pt2, color, 1)
        cv2.circle(out, pt1, 3, color, -1)
        cv2.circle(out, pt2, 3, color, -1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, out)

if __name__ == "__main__":
    path = "data/"
    seq_name = "visages"  # studio, legumes, parc, brain, neige ou magasin

    sys.stdout = open(f"results/superpoint/{seq_name}/log_{seq_name}.txt", "w")

    imagesA = load_images(path, f"{seq_name}A")
    imagesB = load_images(path, f"{seq_name}B")

    superpoint = SuperPoint(max_num_keypoints=1024).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    print("Matching A -> B ...")
    matches_AB, stats_AB = compute_matches(imagesA, imagesB, superpoint, matcher)
    mean_error_AB, errors_AB = compute_error(matches_AB)

    print("Matching B -> A ...")
    matches_BA, stats_BA = compute_matches(imagesB, imagesA, superpoint, matcher)
    mean_error_BA, errors_BA = compute_error(matches_BA)

    print(f"Erreur moyenne A->B : {mean_error_AB}")
    print(f"Erreur moyenne B->A : {mean_error_BA}")
    print(f"Erreur totale : {mean_error_AB + mean_error_BA}")

    # Graphique
    plt.figure(figsize=(10, 4))
    plt.plot(errors_AB, label="A->B")
    plt.plot(errors_BA, label="B->A")
    plt.xlabel("Index image")
    plt.ylabel("Erreur de position")
    plt.title(f"Erreur de localisation avec SuperPoint ({seq_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/superpoint/{seq_name}/erreurs_superpoint_{seq_name}.png")
    plt.show()

    # Sauvegarder A->B
    df_AB = pd.DataFrame(matches_AB, columns=["A_index", "B_index"])
    df_AB["erreur"] = df_AB.apply(lambda row: abs(row["A_index"] - row["B_index"]), axis=1)
    df_AB.to_csv(f"results/superpoint/{seq_name}/superpoint_matches_{seq_name}_A_to_B.csv", index=False)

    # Sauvegarder B->A
    df_BA = pd.DataFrame(matches_BA, columns=["B_index", "A_index"])
    df_BA["erreur"] = df_BA.apply(lambda row: abs(row["B_index"] - row["A_index"]), axis=1)
    df_BA.to_csv(f"results/superpoint/{seq_name}/superpoint_matches_{seq_name}_B_to_A.csv", index=False)

    # Sauvegarde des stats
    pd.DataFrame(stats_AB).to_csv(f"results/superpoint/{seq_name}/superpoint_keypoint_stats_{seq_name}_A_to_B.csv", index=False)
    pd.DataFrame(stats_BA).to_csv(f"results/superpoint/{seq_name}/superpoint_keypoint_stats_{seq_name}_B_to_A.csv", index=False)

    # Exemple visuel sur l’image 3
    idx = 3
    imgA = imagesA[idx][1]
    imgB = imagesB[idx][1]
    featA = extract_features(superpoint, imgA)
    featB = extract_features(superpoint, imgB)
    match_result = match_features(matcher, featA, featB)

    draw_superpoint_matches(imgA, imgB, featA, featB, match_result,
                            f"results/superpoint/{seq_name}/superpoint_matches_{seq_name}_A{idx}_B{idx}.png")


sys.stdout.close()
sys.stdout = sys.__stdout__
print(f"Résultats sauvegardés dans : results/sift/{seq_name}/log_{seq_name}.txt")