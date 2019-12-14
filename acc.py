import numpy as np


def hist_once(a,b,n):
    k = (a >=0) & (a<n)
    return np.bincount(n*a[k].astype(int) + b[k],minlength=n**2).reshape(n,n)



def hist_once2(a,b,n):
    k = (a >=0) & (a<n)
    class_num=np.bincount(a,minlength=11)
    return np.bincount(n*a[k].astype(int) + b[k],minlength=n**2).reshape(n,n),class_num

def acc_eval11(eval_images,labels,num_class=11):
    nlabels= np.argmax(labels,axis=2)
    #hist = np.zeros((num_class,num_class)
    hist,class_num = hist_once2(eval_images.flatten(),nlabels.flatten(),num_class)
    #print(hist)
    # overall accuracy
    # a=float(np.diag(hist).sum())
    # b=float(hist.sum())
    #acc = a / b

    # newhistt=np.delete(hist,[0,1,10],0)
    # newhist=np.delete(newhistt,[0,1,10],1)

    # F-measure
    s0=hist.sum(0)
    s1=hist.sum(1)

    #overall score
    all1 = np.sum(hist[:,1:9])
    all2 = np.sum(hist[1:9,:])
    allr = np.sum(hist[1:9,1:9])
    kall = all1+all2
    acc_all = 2.0*float(allr)/float(kall)

    #brow socre
    b1 = np.sum(hist[1:3,:])
    b2 = np.sum(hist[:,1:3])
    br = np.sum(hist[1:3,1:3])
    acc_brow = 2.0*float(br)/float(b1+b2)
    #print(acc_brow)
    #eye score
    e1 = np.sum(hist[3:5,:])
    e2 = np.sum(hist[:,3:5])
    er = np.sum(hist[3:5,3:5])
    acc_eye = 2.0*float(er)/float(e1+e2)
    #print(acc_eye)



    #mouth score
    m1 = np.sum(hist[:,6:9])
    m2 = np.sum(hist[6:9,:])
    mr = np.sum(hist[6:9,6:9])
    km = m1+m2
    acc_mouth = 2.0*float(mr)/float(km)

    #face socre
    f1 = np.sum(hist[:,9])
    f2 = np.sum(hist[9,:])
    fr = np.sum(hist[9,9])
    fm = f1+f2
    acc_face = 2.0*float(fr)/float(fm)

    #bg
    k0=float(s0[0]+s1[0])
    if k0==0:
        acc_bg=1.0
    else:
        acc_bg=float(2*hist[0][0])/k0

    #b_l
    k1 = float(s0[1] + s1[1])
    if k1 == 0:
        acc_bl = 1.0
    else:
        acc_bl = float(2 * hist[1][1]) / k1

    #b_r
    k2 = float(s0[2] + s1[2])
    if k2 == 0:
        acc_br = 1.0
    else:
        acc_br = float(2 * hist[2][2] )/ k2

    #e_l
    k3 = float(s0[3] + s1[3])
    if k3 == 0:
        acc_el = 1.0
    else:
        acc_el = float(2 * hist[3][3] )/ k3

    #e_r
    k4 =float( s0[4] + s1[4])
    if k4 == 0:
        acc_er = 1.0
    else:
        acc_er = float(2 * hist[4][4]) / k4

    #nose
    k5 = float(s0[5] + s1[5])
    if k5 == 0:
        acc_nose = 1.0
    else:
        acc_nose = float(2 * hist[5][5]) / k5

    # lip_up
    k6 = float(s0[6] + s1[6])
    if k6 == 0:
        acc_lup = 1.0
    else:
        acc_lup =float( 2 * hist[6][6] )/ k6

    # mouth_in
    k7 = float(s0[7] + s1[7])
    if k7 == 0:
        acc_mi = 1.0
    else:
        acc_mi = float(2 * hist[7][7] )/ k7

    # lip_l
    k8 = float(s0[8] + s1[8])
    if k8 == 0:
        acc_ll = 1.0
    else:
        acc_ll = float(2 * hist[8][8]) / k8



    return acc_all,acc_bg,acc_bl,acc_br,acc_el,acc_er,acc_nose,acc_lup,acc_mi,acc_ll,acc_mouth,acc_brow,acc_eye,acc_face


def acc_eval(eval_images,labels,num_class=10):
    nlabels= np.argmax(labels,axis=2)
    #hist = np.zeros((num_class,num_class)
    hist,class_num = hist_once2(eval_images.flatten(),nlabels.flatten(),num_class)
    #print(hist)
    # overall accuracy
    # a=float(np.diag(hist).sum())
    # b=float(hist.sum())
    #acc = a / b

    # newhistt=np.delete(hist,[0,1,10],0)
    # newhist=np.delete(newhistt,[0,1,10],1)

    # F-measure
    s0=hist.sum(0)
    s1=hist.sum(1)

    #overall score
    all1 = np.sum(hist[0:2,2:10])
    all2 = np.sum(hist[2:10,0:2])
    allr = np.sum(hist[2:10,2:10])
    kall = all1+all2+2*allr
    acc_all = 2.0*float(allr)/float(kall)

    #brow socre
    b1 = np.sum(hist[2:4,:])
    b2 = np.sum(hist[:,2:4])
    br = np.sum(hist[2:4,2:4])
    acc_brow = 2.0*float(br)/float(b1+b2)
    #print(acc_brow)
    #eye score
    e1 = np.sum(hist[4:6,:])
    e2 = np.sum(hist[:,4:6])
    er = np.sum(hist[4:6,4:6])
    acc_eye = 2.0*float(er)/float(e1+e2)
    #print(acc_eye)



    #mouth score
    m1 = np.sum(hist[0:7,7:10])
    m2 = np.sum(hist[7:10,0:7])
    mr = np.sum(hist[7:10,7:10])
    km = m1+m2+2*mr
    acc_mouth = 2.0*float(mr)/float(km)


    k0=float(s0[0]+s1[0])
    if k0==0:
        acct0=1.0
    else:
        acct0=float(2*hist[0][0])/k0

    k1 = float(s0[1] + s1[1])
    if k1 == 0:
        acct1 = 1.0
    else:
        acct1 = float(2 * hist[1][1]) / k1

    k2 = float(s0[2] + s1[2])
    if k2 == 0:
        acct2 = 1.0
    else:
        acct2 = float(2 * hist[2][2] )/ k2

    k3 = float(s0[3] + s1[3])
    if k3 == 0:
        acct3 = 1.0
    else:
        acct3 = float(2 * hist[3][3] )/ k3

    k4 =float( s0[4] + s1[4])
    if k4 == 0:
        acct4 = 1.0
    else:
        acct4 = float(2 * hist[4][4]) / k4

    k5 = float(s0[5] + s1[5])
    if k5 == 0:
        acct5 = 1.0
    else:
        acct5 = float(2 * hist[5][5]) / k5

    k6 = float(s0[6] + s1[6])
    if k6 == 0:
        acct6 = 1.0
    else:
        acct6 =float( 2 * hist[6][6] )/ k6

    k7 = float(s0[7] + s1[7])
    if k7 == 0:
        acct7 = 1.0
    else:
        acct7 = float(2 * hist[7][7] )/ k7

    k8 = float(s0[8] + s1[8])
    if k8 == 0:
        acct8 = 1.0
    else:
        acct8 = float(2 * hist[8][8]) / k8

    k9 = float(s0[9] + s1[9])
    if k9 == 0:
        acct9 = 1.0
    else:
        acct9 = float(2 * hist[9][9]) / k9





    return acc_all,acct0,acct1,acct2,acct3,acct4,acct5,acct6,acct7,acct8,acct9,acc_mouth,acc_brow,acc_eye

