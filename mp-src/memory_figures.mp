prologues:=3;
outputformat:="png";
outputtemplate:=("%j_%c" & "." & outputformat);
hppp:=0.1;
vppp:=0.1;

verbatimtex
%&latex
\documentclass{minimal}
\usepackage{amssymb}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{mathrsfs}
\begin{document}
etex

input hatching;

vardef box(expr w, h) = 
    unitsquare xscaled w yscaled h shifted (-w/2, -h/2)
enddef;

beginfig(0);
    u:=30;

    path tempbox;
    pair ptemp, porigin;

    % ram

    draw box(3u, 2u);
    label(btex RAM etex, origin);

    % l3

    z1 = (0, -3u);
    tempbox := box(8.5u, 2u);
    draw tempbox shifted z1;

    hatchfill tempbox shifted z1 withcolor (45, 2mm, -.2bp);

    tempbox := box(4.5u, 1.5u);
    fill tempbox shifted z1 withcolor white;
    draw tempbox shifted z1;
    label(btex L3 Cache etex, z1);

    % arrow ram - l3

    drawdblarrow (origin+1u*down)--(z1+0.75u*up);

    % l2
    tempbox := box(8.5u, 6u);
    z2 = z1 + (0, -4.5u);
    draw tempbox shifted z2;
    hatchfill tempbox shifted z2 withcolor (45, 2mm, -.2bp);

    z3 = z2 + (0, 1.7u);
    
    tempbox := box(4.5u, 1.5u);
    fill tempbox shifted z3 withcolor white;
    draw tempbox shifted z3;

    label(btex L2 Cache etex, z3);

    tempbox := box(1.5u, 1u);

    z4 = z3 + 2.2u*down;

    % arrow l3 - l2

    drawdblarrow (z1 + 0.75u*down)--(z3 + 0.75u*up);

    % l1 caches

    for i=0 upto 3:
        ptemp := z4 + i*1.8u*right + 2.7u*left;
        fill tempbox shifted ptemp withcolor white;
        draw tempbox shifted ptemp;
        
        porigin := (z3+0.75u*down) + i*0.8u*right + 1.2u*left;
        drawdblarrow porigin--(ptemp + 0.5u*up);
        
        label.top(btex L1 etex, ptemp);
        if odd(i):
            label.bot(btex ICache etex, ptemp);
        else:
            label.bot(btex DCache etex, ptemp);
        fi;
    endfor;

    % cores

    z5 = z4 + 1.4u*down;
    tempbox := box(3.3u, 1u);
    for i=0 upto 1:
        ptemp := z5 + i*3.6u*right + 1.8u*left;
        fill tempbox shifted ptemp withcolor white;
        draw tempbox shifted ptemp;
        label(btex Core etex, ptemp);

        ptemp := ptemp + 0.85u*left + 0.5u*up;
        porigin := (xpart ptemp, y4 - 0.5u);
        drawdblarrow porigin--ptemp;
        ptemp := ptemp + 1.75u*right;
        porigin := porigin + 1.75u*right;
        drawdblarrow porigin--ptemp;

    endfor;

endfig;

end;