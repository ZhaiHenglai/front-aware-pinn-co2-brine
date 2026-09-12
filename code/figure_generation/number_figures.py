"""Vector-preserving figure composition and numbered asset copies."""
import json,shutil,hashlib
import fitz
from PIL import Image
from shared import ROOT,OUT
dest=ROOT/'02_figures/numbered';main=dest/'main';supp=dest/'supplement'
main.mkdir(parents=True,exist_ok=True);supp.mkdir(exist_ok=True)
mm=72/25.4
manifest=[]
def copy(source,target):
    for ext in ['pdf','svg','png','tiff']:
        a=source.with_suffix('.'+ext);b=target.with_suffix('.'+ext);shutil.copy2(a,b)
        assert hashlib.sha256(a.read_bytes()).digest()==hashlib.sha256(b.read_bytes()).digest()
    manifest.append({'destination':str(target.relative_to(ROOT)),'source':str(source.relative_to(ROOT)),'operation':'byte-identical copy'})
def compose(name,top,bottom,bottom_width,label=None):
    w=190*mm;gap=8*mm
    with fitz.open(top) as a,fitz.open(bottom) as b:
        th=a[0].rect.height*w/a[0].rect.width;bw=bottom_width*mm;bh=b[0].rect.height*bw/b[0].rect.width
        d=fitz.open();p=d.new_page(width=w,height=th+gap+bh+2*mm)
        p.show_pdf_page(fitz.Rect(0,0,w,th),a,0)
        left=(w-bw)/2;p.show_pdf_page(fitz.Rect(left,th+gap,left+bw,th+gap+bh),b,0)
        if label:p.insert_text((left-5*mm,th+gap+9),label,fontsize=10,fontname='tibo')
        target=main/name;d.save(target.with_suffix('.pdf'))
        target.with_suffix('.svg').write_text(p.get_svg_image(text_as_path=False))
        pix=p.get_pixmap(dpi=600,alpha=False)
        im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
        im.save(target.with_suffix('.png'),dpi=(600,600));im.save(target.with_suffix('.tiff'),dpi=(600,600),compression='tiff_lzw')
        p.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(main/f'{name}_preview.png')
        manifest.append({'destination':str(target.relative_to(ROOT)),'sources':[str(x.relative_to(ROOT)) for x in [top,bottom]],'operation':'full vector page composition','width_mm':190,'height_mm':p.rect.height/mm})
        d.close()
compose('Figure_1',OUT/'Figure1.pdf',OUT/'Figure3.pdf',100,'d')
compose('Figure_5',OUT/'Figure9.pdf',OUT/'Front_ray_panels_cd.pdf',170)
for n,source in {2:ROOT/'02_figures/revised/Method_legacy_refined',3:OUT/'Figure6',4:OUT/'Figure8',6:OUT/'H_FV_tradeoff',7:OUT/'K_fields_main',8:OUT/'HK_component_comparison'}.items():copy(source,main/f'Figure_{n}')
copy(OUT/'Figure2',supp/'Figure_A1')
for n,source in {1:OUT/'FigureB1',2:OUT/'FigureB2',3:OUT/'FigureB3',4:OUT/'FigureB4',5:ROOT/'02_figures/revised/H_ablation',6:OUT/'Figure5',7:OUT/'Figure7',8:OUT/'Figure11'}.items():copy(source,supp/f'Figure_B{n}')
copy(OUT/'K_fields_complete',supp/'Figure_C1');copy(OUT/'graphical_abstract',dest/'graphical_abstract')
book=fitz.open();toc=[]
paths=[(f'Figure {n}',main/f'Figure_{n}.pdf') for n in range(1,9)]+[('Figure A.1',supp/'Figure_A1.pdf')]+[(f'Figure B.{n}',supp/f'Figure_B{n}.pdf') for n in range(1,9)]+[('Figure C.1',supp/'Figure_C1.pdf'),('Graphical abstract',dest/'graphical_abstract.pdf')]
for title,path in paths:
    with fitz.open(path) as d:toc.append([1,title,len(book)+1]);book.insert_pdf(d)
book.set_toc(toc);book.save(dest/'NUMBERED_FIGURES_REVIEW.pdf');book.close()
(ROOT/'04_checks/BATCH25_ASSET_MAP.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('19 figure pages; 8 main figures; source variants retained.')
