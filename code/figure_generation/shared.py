"""Common Python export and source-preservation helpers."""
from pathlib import Path
import hashlib
import os,json,shutil
import matplotlib as mpl
import fitz
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'02_figures/legacy_refined'
SRC=ROOT/'06_evidence/figure_revision_all/source_data'
H=Path(os.environ.get('PINN_H_ASSET_ROOT', ROOT/'external_assets/H'))
FW=H/'Results2/full_workflow'
mpl.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42})
def save(fig,name):
    fig.canvas.draw()
    for ext in ['pdf','svg','png','tiff']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.04)
    with fitz.open(OUT/f'{name}.pdf') as doc:
        page=doc[0];page.get_pixmap(matrix=fitz.Matrix(1.7,1.7)).save(OUT/f'{name}_preview.png')
        (OUT/f'{name}_export.json').write_text(json.dumps({'width_mm':page.rect.width*25.4/72,
            'height_mm':page.rect.height*25.4/72,'text_chars':len(page.get_text()),'dpi':600},indent=2))
def capture(path):
    path=Path(path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
    dest=SRC/f'{digest[:12]}_{path.name}'
    if not dest.exists():shutil.copy2(path,dest)
    assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
    return dest
