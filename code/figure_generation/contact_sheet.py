"""Python-only review sheets; inspect individual exports for dense details."""
from PIL import Image,ImageOps,ImageDraw
from shared import OUT
groups=[['Figure1','Figure2','Figure3','Figure5'],['Figure6','Figure7','Figure8'],['Figure9','Figure10','Figure11'],['FigureB1','FigureB2','FigureB3','graphical_abstract']]
for i,names in enumerate(groups):
    canvas=Image.new('RGB',(1600,500*len(names)),'white');d=ImageDraw.Draw(canvas)
    for j,name in enumerate(names):
        img=Image.open(OUT/f'{name}_preview.png').convert('RGB')
        img=ImageOps.contain(img,(1550,460))
        canvas.paste(img,((1600-img.width)//2,j*500+30))
        d.text((12,j*500+8),name,fill='black')
    canvas.save(OUT/f'contact_{i+1}.png')
