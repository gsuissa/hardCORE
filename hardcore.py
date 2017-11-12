import numpy as np

# name = hardcore
# authors = Gabrielle Suissa & David Kipping
# url = https://github.com/gsuissa/hardCORE
# license = GNU GPLv3
# description = solves for core radius fractions using exoplanet mass & radius alone
# requirements = numpy
# reference and citation: Suissa, G., Chen, J. & Kipping, D. 2018, MNRAS, submitted


def forward(Mobs, crf):
    """
    Returns the radius [earth radii] of a solid planet given
    its observed mass and minumum core radius fraction (CRFmin).

    Note:
        This is the forward model. For a more practical approach, see
        hardcore.invert

    Args:
        M_obs (float/int): mass [earth masses], recommended for >0.1
        CRF (float/int): core radius fraction, 0 <= CRF <= 1

    Returns:
        float
    """
    a0 = (-0.40660382*crf**5 + 1.43688143*crf**4 - 1.52574933*crf**3 +
          0.24692514*crf**2 - 0.02281595*crf + 1.04285919)
    a1 = (-0.60684066947724757*crf**5 + 1.8276267952863359*crf**4 -
          1.7038407459165059*crf**3 + 0.29220353126558546*crf**2 -
          0.024719182117647742*crf + 0.7172980715863525)
    a2 = (0.88476320909079476*crf**7 - 2.9516123779638179*crf**6 +
          3.4071221416329145*crf**5 - 1.3463934835325109*crf**4 -
          0.076864161485612648*crf**3 +
          0.012755987597017642*crf**2 - 0.001194168411679495*crf +
          0.20320064867549339)
    a3 = (7.3586169293345218*crf**10 - 34.234323353588991*crf**9 +
          66.02343215572337*crf**8 - 67.898807477736085*crf**7 +
          39.83901992398571*crf**6 - 13.375868926380381*crf**5 +
          2.6424705878154442*crf**4 - 0.3784861937084899*crf**3 +
          0.017227789730960362*crf**2 - 0.00033751543297382214*crf +
          0.015508503902881452)
    a4 = (-0.16544738405664638*crf**7 + 0.61484443762522456*crf**6 -
          0.88699599239885818*crf**5 + 0.59947843282735935*crf**4 -
          0.17549161407586755*crf**3 +
          0.018847420317995751*crf**2 - 0.0011290421327988804*crf -
          0.0098270586400556192)
    a5 = (-0.010645546128882312*crf**7 +
          0.028878834664370984*crf**6 - 0.0094642012405933857*crf**5 -
          0.033669736474959695*crf**4 +
          0.031419614235720324*crf**3 - 0.0045195837580368804*crf**2 +
          0.00023752295782753652*crf - 0.0044109138512083311)
    a6 = (-0.39256426792356286*crf**10 + 1.7883663418218612*crf**9 -
          3.3879401501872337*crf**8 + 3.4490513000309346*crf**7 -
          2.0380509469731902*crf**6 +
          0.71217205567569064*crf**5 - 0.15098021323361266*crf**4 +
          0.021119477115077365*crf**3 + 0.00072694660016693461*crf**2 +
          2.5488168497226274e-5*crf - 0.00081113972015464219)
    a7 = (-0.030096287375531935*crf**10 +
          0.13520827069266955*crf**9 - 0.25133568444817084*crf**8 +
          0.24949342651821868*crf**7 - 0.14287301624904816*crf**6 +
          0.048466462452411284*crf**5 - 0.010295574934109469*crf**4 +
          0.0014697735055552036*crf**3 - 1.6931093281375233e-6*crf**2 +
          1.6532360682938199e-6*crf - 5.7559003900993822e-5)

    Robs = (a0 + a1*(np.log10(Mobs))**1 + a2*(np.log10(Mobs))**2 +
            a3*(np.log10(Mobs))**3 + a4*(np.log10(Mobs))**4 +
            a5*(np.log10(Mobs))**5 + a6*(np.log10(Mobs))**6 +
            a7*(np.log10(Mobs))**7)

    return Robs


def newtonfn(Mobs, Robs, crf):
    """
    Returns the f(x)/f'(x) term in Newton's method used in hardcore.invert

    Args:
        M_obs (float/int): mass [earth masses], recommended for >0.1
        R_obs (float/int): radius [earth radii]
        crf (float/int): core radius fraction, 0 <= CRF <= 1

    Returns:
        float
    """
    x = ((2.4316961243316095 - 0.053201292868641066*crf +
          0.5757698754498582*crf**2 - 3.5576794720124685*crf**3 +
          3.3504609615177876*crf**4 - 0.9481020474417332*crf**5 -
          2.3317588296187997*Robs + (0.7263862330122107 -
          0.0250323739779068*crf + 0.29590574791238944*crf**2 -
          1.72542839595521*crf**3 + 1.8507828136833355*crf**4 -
          0.6145293364100723*crf**5)*np.log(Mobs) + (0.0893670336283766 -
          0.0005251916728619044*crf + 0.00561004494806671*crf**2 -
          0.03380462685073185*crf**3 - 0.5921398012465516*crf**4 +
          1.498442062030764*crf**5 - 1.2981102391098687*crf**6 +
          0.389116196111353*crf**7)*np.log(Mobs)**2 +
          (0.002962146048249629 - 0.00006446592219775078*crf +
          0.0032905320584894243*crf**2 - 0.07229139509726291*crf**3 +
          0.5047155972188316*crf**4 - 2.5548097695497947*crf**5 +
          7.60930881360172*crf**6 - 12.968767684526258*crf**7 +
          12.610568361504987*crf**8 - 6.538803889232176*crf**9 +
          1.405506178694721*crf**10)*np.log(Mobs)**3 +
          (-0.0008151629320623728 - 0.00009365501205445961*crf +
          0.0015634096600997188*crf**2 - 0.014557179713912134*crf**3 +
          0.049727249516946063*crf**4 - 0.07357707737127488*crf**5 +
          0.051001872777466414*crf**6 - 0.01372400222991872*crf**7) *
          np.log(Mobs)**4 + (-0.00015890360945395335 +
          8.556788139635795e-6*crf - 0.00016281845363739863*crf**2 +
          0.0011318947225276395*crf**3 - 0.0012129555996131895*crf**4 -
          0.00034094878940265256*crf**5 + 0.0010403628861931627*crf**6 -
          0.00038350685630021014*crf**7)*np.log(Mobs)**5 +
          (-0.000012690687977777016 + 3.9877518692049377e-7*crf +
          0.000011373444364758243*crf**2 + 0.00033042481789715783*crf**3 -
          0.0023621612027589136*crf**4 + 0.01114228920185189*crf**5 -
          0.0318863298248035*crf**6 + 0.05396213843367694*crf**7 -
          0.05300602382683492*crf**8 + 0.027979888877458607*crf**9 -
          0.006141864973019203*crf**10)*np.log(Mobs)**6 +
          (-3.910993360147583e-7 + 1.1233334226866258e-8*crf -
          1.1504263263033043e-8*crf**2 + 9.986751040783711e-6*crf**3 -
          0.00006995591041753379*crf**4 + 0.000329317743523268*crf**5 -
          0.0009707871575669106*crf**6 + 0.0016952467353181114*crf**7 -
          0.0017077644268058158*crf**8 + 0.0009187070885132835*crf**9 -
          0.0002044969025059285*crf**10)*np.log(Mobs)**7) /
          (-0.10640258573728215 + 2.3030795017994334*crf -
          21.34607683207481*crf**2 + 26.8036876921423*crf**3 -
          9.481020474417333*crf**4 + (-0.05006474795581361 +
          1.1836229916495575*crf - 10.352570375731263*crf**2 +
          14.806262509466686*crf**3 - 6.145293364100724*crf**4)*np.log(Mobs) +
          (-0.0010503833457238087 + 0.022440179792266843*crf -
          0.20282776110439113*crf**2 - 4.737118409972412*crf**3 +
          14.984420620307638*crf**4 - 15.577322869318422*crf**5 +
          5.447626745558943*crf**6)*np.log(Mobs)**2 +
          (-0.00012893184439550156 + 0.013162128233957697*crf -
          0.43374837058357757*crf**2 + 4.037724777750653*crf**3 -
          25.548097695497948*crf**4 + 91.31170576322064*crf**5 -
          181.56274758336758*crf**6 + 201.76909378407984*crf**7 -
          117.69847000617916*crf**8 + 28.11012357389442*crf**9) *
          np.log(Mobs)**3 + (-0.00018731002410891923 +
          0.006253638640398875*crf - 0.08734307828347279*crf**2 +
          0.3978179961355687*crf**3 - 0.7357707737127488*crf**4 +
          0.612022473329597*crf**5 - 0.1921360312188621*crf**6) *
          np.log(Mobs)**4 + (0.000017113576279271593 -
          0.0006512738145495947*crf + 0.006791368335165838*crf**2 -
          0.009703644796905516*crf**3 - 0.0034094878940265257*crf**4 +
          0.012484354634317952*crf**5 - 0.005369095988202942*crf**6) *
          np.log(Mobs)**5 + (7.975503738409874e-7 +
          0.00004549377745903297*crf + 0.0019825489073829468*crf**2 -
          0.018897289622071316*crf**3 + 0.11142289201851888*crf**4 -
          0.38263595789764204*crf**5 + 0.7554699380714772*crf**6 -
          0.8480963812293592*crf**7 + 0.5036379997942549*crf**8 -
          0.12283729946038406*crf**9)*np.log(Mobs)**6 +
          (2.246666845373252e-8 - 4.601705305213217e-8*crf +
          0.00005992050624470227*crf**2 - 0.0005596472833402703*crf**3 +
          0.00329317743523268*crf**4 - 0.011649445890802928*crf**5 +
          0.02373345429445356*crf**6 - 0.027324230828893053*crf**7 +
          0.016536727593239105*crf**8 -
          0.0040899380501185694*crf**9)*np.log(Mobs)**7))

    return x


def Rcoreless(Mobs):
    """
    Returns the radius of a solid planet if it was to have no iron core,
    given its mass.

    Args:
        M_obs (float/int): mass [earth masses], recommended for >0.1

    Returns:
        float
    """
    x = (1.0424733556465293 +
         3.113559991697741e-1*(np.log(Mobs)) +
         3.8322380977759925e-2*(np.log(Mobs))**2 +
         1.2703215611136627e-3*(np.log(Mobs))**3 -
         3.499043718139537e-4*(np.log(Mobs))**4 -
         6.811637077809525e-5*(np.log(Mobs))**5 -
         5.442248192483103e-6*(np.log(Mobs))**6 -
         1.6771941752060635e-7*(np.log(Mobs))**7)

    return x


def Rcorefull(Mobs):
    """
    Returns the radius of a solid planet if it was made of 100% iron.

    Args:
        M_obs (float/int): mass [earth masses], recommended for >0.1

    Returns:
        float
    """
    x = (7.714041890116314e-1 +
         2.179663002059768e-1*(np.log(Mobs)) +
         2.4851443723223846e-2*(np.log(Mobs))**2 +
         6.91594560060839e-4*(np.log(Mobs))**3 -
         2.0372392728182706e-4*(np.log(Mobs))**4 -
         3.36603267896379e-5*(np.log(Mobs))**5 -
         2.2709431294462362e-6*(np.log(Mobs))**6 -
         5.881866013051586e-8*(np.log(Mobs))**7)

    return x


def CRFmax(Mobs, Robs):
    """
    Returns the maximum core radius fraction (CRFmax) of a solid planet given 
    its observed mass and radius.

    Args:
        M_obs (float/int): mass [earth masses], recommended for >0.1
        R_obs (float/int): radius [earth radii]

    Returns:
        float
    """
    x = Rcorefull(Mobs)/Robs

    return x


def invert(M_test, R_test):
    """
    Returns:
        The minimum core radius fraction (CRFmin) of a solid planet given
            its observed mass and radius.
        The maximum core radius fraction (CRFmax) of a solid planet from 
            hardcore.CRFmax
        The marginal core radius fraction (CRFmarg) of a solid planet as
            a random sample from uniform distribution
            between CRFmin and CRFmax.

    Args:
        M_test (float/int): mass [earth masses], recommended for >0.1
        R_test (float/int): radius [earth radii]

    Returns:
        tuple
    """
    tol = 1.0e-3
    jmin = 3
    jmax = 50

    # CRFmin
    if R_test <= Rcorefull(M_test):
        CRFmin_test = 1.0
    elif R_test >= Rcoreless(M_test):
        CRFmin_test = 0.0
    else:
        # solve iteratively
        xold = 0.0
        xnew = 0.5
        j = 1
        while (abs(xnew-xold) > tol or j < jmin) and j < jmax:
            j = j + 1
            xold = xnew
            xnew = xold - newtonfn(M_test, R_test, xold)
        CRFmin_test = min(max(xnew, 0.0), 1.0)

    # CRFmax
    CRFmax_test = min(max(CRFmax(M_test, R_test), 0.0), 1.0)

    # CRFmarg
    CRFmarg_test = np.random.uniform(CRFmin_test, CRFmax_test)

    return (CRFmin_test, CRFmax_test, CRFmarg_test)
