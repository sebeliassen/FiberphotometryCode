
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Pathing
# Pathing
this_file = Path(__file__).resolve()
repo_root = this_file.parents[5]
datta_dir = this_file.parents[1]

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from lib import config

def create_side_by_side(img1_path, img2_path, out_path, label1="IFDN (Baseline)", label2="AMPH (Saline)"):
    try:
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # Ensure same height for side-by-side
        if img1.size[1] != img2.size[1]:
            new_h = img1.size[1]
            img2 = img2.resize((int(img2.size[0] * (new_h / img2.size[1])), new_h))
            
        w1, h1 = img1.size
        w2, h2 = img2.size
        
        header_h = 40
        combined = Image.new('RGB', (w1 + w2, h1 + header_h), color=(255, 255, 255))
        
        combined.paste(img1, (0, header_h))
        combined.paste(img2, (w1, header_h))
        
        draw = ImageDraw.Draw(combined)
        draw.text((10, 10), label1, fill="black")
        draw.text((w1 + 10, 10), label2, fill="black")
        
        combined.save(out_path)
        return True
    except Exception as e:
        print(f"Error creating comparison: {e}")
        return False

def main():
    ifdn_dir = config.OUTPUT_ROOT / "panel_1f_ifdn" / "plots_unified_ci"
    amph_dir = config.OUTPUT_ROOT / "panel_1f_amph" / "plots_unified_ci"
    out_dir = config.OUTPUT_ROOT / "cross_project_comparison_1f"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not ifdn_dir.exists() or not amph_dir.exists():
        print(f"Directories missing: IFDN({ifdn_dir.exists()}), AMPH({amph_dir.exists()})")
        return

    ifdn_plots = sorted(list(ifdn_dir.glob("ifdn_1f_*.png")))
    print(f"Found {len(ifdn_plots)} IFDN plots.")
    
    count = 0
    for ip in ifdn_plots:
        # Reconstruct AMPH filename
        # IFDN: ifdn_1f_KI_baseline_DLS_Avg_Datta_Z-Score.png
        # AMPH: amph_1f_KI_Saline_DLS_Avg_Datta_Z-Score.png
        
        name = ip.name
        amph_name = name.replace("ifdn_", "amph_").replace("_baseline_", "_Saline_")
        ap = amph_dir / amph_name
        
        if ap.exists():
            out_name = f"Compare_{name.replace('ifdn_', '')}"
            if create_side_by_side(ip, ap, out_dir / out_name):
                count += 1
                
    print(f"Created {count} cross-project comparisons in {out_dir}")

if __name__ == "__main__":
    main()
