# validate_widerface_annotations.py

# =============================================================================
# FACE PREPROCESSING / ALIGNMENT (annotation validation)
# =============================================================================

annotation_path = "C:\\coding\\transformer\\ESRGAN_Complete\\wider_face\\wider_face_split\\wider_face_train_bbx_gt.txt"

with open(annotation_path, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

i = 0
error_found = False
while i < len(lines):
    if i + 1 >= len(lines):
        print(f"❌ Unexpected EOF at line {i + 1}")
        error_found = True
        break

    img_path = lines[i]
    next_line = lines[i + 1]

    if not next_line.isdigit():
        print(f"❌ Line {i + 2}: Expected number of faces, got '{next_line}' (after image '{img_path}')")
        error_found = True
        break

    num_faces = int(next_line)
    i += 2 + num_faces

if not error_found:
    print("✅ All annotations appear to be correctly formatted.")
