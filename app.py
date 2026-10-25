# NACA Airfoil Simulator

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from scipy.integrate import solve_ivp
# Numpy increases code efficiency

# Modelling the Airfoil
    # Parameters
        # m = maximum camber
        # p = location of maximum camber
        # t = maximum thickness
        # x = position along the chord
        # yc = positions along the mean camber line
        # C = Chord Length

        # all parameters are expressed as a fraction of the chord length C; C = 1.0

def CreateAirfoil(mycode, n):
    
    # mycode = 4 digit NACA code
    # n = discretization of the airfoil (number of points along the chord)

    mycode = mycode.strip()  # Remove any whitespace from the input code
    if len(mycode) != 4 or not mycode.isdigit():
        raise ValueError("You have not provided a valid code")

    # isdigit() returns True if all characters in the string are digits
    # raise halts program execution and displays an error message. 

    m = int(mycode[0]) / 100
    p = int(mycode[1]) / 10
    t = int(mycode[2:]) / 100

    # 10% =< p <= 90%; Hence divide by 10. Or, p% * 10

    # Cosine spacing
        # Instead of spreading out points linearly at regular intervals, each x parameter is treated as an angle of which the cosine is taken.
        # Since cosine(theta) changes slowly between theta = 0 and pi/2, you get more points cluttered near the leading edge. This helps generate a smoother curve for the airfoil.
        
    alpha = np.linspace(0.0, np.pi, n + 1)
    # np.linspace(start, stop, number) gives number amount of points between start and stop inclusive.
    # We need n+1 discrete points to create n panels.
    x = 0.5 * (1 - np.cos(alpha))
    # algebraic modification so that 0 =< x <= 1.0

    # Application of NACA Formulae
    # Thickness
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)

    # Theta
    if p == 0.0:
        yc = np.zeros_like(x) # "give me another array with the same dimensions as x full of zeros"
        dyc = np.zeros_like(x)
    else:
        yc = np.where(x < p, m/p**2*(2*p*x-x**2), m/(1-p)**2*((1-2*p)+2*p*x - x**2))
        dyc = np.where(x < p, 2*m/p**2*(p-x), 2*m/(1-p)**2*(p-x))

    theta = np.arctan(dyc)
    
    # Surface Coordinates
    xupper = x  - yt * np.sin(theta)
    yupper = yc + yt * np.cos(theta)
    xlower = x  + yt * np.sin(theta)
    ylower = yc - yt * np.cos(theta)

    return xupper, yupper, xlower, ylower

# Using the Vortex Panel Method

def CreatePanels(xupper, yupper, xlower, ylower):
    # Logically, we are concatenating the n+1 points in a closed loop to give us n panels.
    # We begin at the trailing edge, move upwards towards the leading edge and loop back around to the trailing edge
    xpanel =  np.concatenate([xupper[::-1], xlower[1:]])
    ypanel =  np.concatenate([yupper[::-1], ylower[1:]])
    return xpanel, ypanel

# Lets Apply the Vortex Panel Method
# Firstly the Boundary Condition. We ask ourselves: How much of total velocity of air flowing along a panel is in the directionnormal to the panel? We want the answer to be zero i.e. all flow is tangential
# Mathematically, Vtotal * normal = 0 (dot product)

def SolveEquations(xpanel, ypanel, aoa):
    
    # Convert angle of attack into radians
    aoa = np.radians(aoa)

    # Calculate Number of panels as number of discretization points - 1
    N = len(xpanel) - 1

    # Find the position of the collocation or control points. They're Midpoints
    xcontrol = 0.5 * (xpanel[:-1] + xpanel[1:])
    # The idea is to find the midpoints for each panel. First we generate two sliced arrays, one with all elements other than the last one; the other with all elements other than the first one. Then, numpy does the looping part for you. 
    ycontrol = 0.5 * (ypanel[:-1] + ypanel[1:])

    # Calculating length of the panels
    xcomponent = xpanel[1:] - xpanel[:-1] #This logically is x2 - x1
    ycomponent = ypanel[1:] - ypanel[:-1] 
    PanelLength = np.hypot(xcomponent,ycomponent)

    # Now let us find the direction normal to the panel. This is vital for the boundary condition
    costheta = xcomponent/PanelLength
    sintheta = ycomponent/PanelLength
    # The resultant of these two ratios is a unit vector in the direction of L. Since we're concerned with direction and not magnitude, we're normalising everything to unit vectors

    normal_xcomponent = -sintheta
    normal_ycomponent = costheta

    # Influence Matrix A, each entry aij represents the effect of panel j on control panel i
    A = np.zeros((N,N))

    # A bunch of formula work to calculate each entry aij
    for i in range(N):
        # Start --> "S"
        x_S = xcontrol[i] - xpanel[:-1] # x component of distance from start of each panel to the control point
        y_S = ycontrol[i] - ypanel[:-1] # y component of distance from start of each panel to the control point
        # End --> "E"
        x_E = xcontrol[i] - xpanel[1:] # x component of distance from end of each panel to the control point
        y_E = ycontrol[i] - ypanel[1:] # y component of distance from start of each panel to the control point

        distance_S = (x_S)**2 + (y_S)**2
        distance_E = (x_E)**2 + (y_E)**2

        # The following formula are found by integrating the formulae for point entries to form panels
        logdistance = 0.5 * np.log(distance_E / distance_S)
        # The ratio gives us viewing perspective. A large value would mean the control panel was closer to the end of a panel --> stronger influence by the influence panel
        crossprod = x_S * y_E - y_S * x_E
        dotprod = x_S * x_E + y_S * y_E
        angleatcontrol = np.arctan2(crossprod , dotprod)
        # the angleatcontrol is the angle formed at the control point by the two distance vectors. depending on how wide the angle is, you can estimate how much influence is caused.

        u = (1.0 / (2 * np.pi)) * (sintheta * logdistance + costheta * angleatcontrol)
        v = (1.0 / (2 * np.pi)) * (-costheta * logdistance + sintheta * angleatcontrol)

        A[i, :] = u * normal_xcomponent + v * normal_ycomponent
        A[i, i] = 0.5

    # Now we calculate the RHS
    rhs = -(np.cos(aoa) * normal_xcomponent + np.sin(aoa) * normal_ycomponent)

    # Now we use the Kutta Condition
    A[-1, :] = 0.0
    A[-1, 0] = 1.0
    A[-1, -1] = 1.0
    rhs[-1] = 0.0

    # Now we solve for the strength (gamma) values

    gamma = np.linalg.solve(A,rhs)
    # Telling python to solve for such a solution gamma that A equals rhs. Exactly what the formula says!

    # Now we'll be finding velocities at any arbitrary point so that we can use them to trace streamlines!

    def ArbitraryV(xV, yV):
        x_Start = xV - xpanel[:-1]
        y_Start = yV - ypanel[:-1]
        x_End = xV - xpanel[1:]
        y_End = yV - ypanel[1:]

        distance_Start = (x_Start)**2 + (y_Start)**2
        distance_End = (x_End)**2 + (y_End)**2

        distancelog = 0.5 * np.log(distance_End / distance_Start)
        beta2 = np.arctan2(x_Start * y_End - y_Start * x_End, x_Start * x_End + y_Start * y_End)

        u = (1.0 / (2 * np.pi)) * (sintheta * distancelog + costheta * beta2)
        v = (1.0 / (2 * np.pi)) * (-costheta * distancelog + sintheta * beta2)

        u1 = np.cos(aoa) + np.dot(gamma, u)
        v1 = np.sin(aoa) + np.dot(gamma, v)

        return u1, v1

    return ArbitraryV

def CreateStreamlines(ArbitraryV, AirfoilBoundary, NumLines = 50, ylow = -0.65, yhigh = 0.65, xstart = -0.55, tinteg = 8.0, numpoints = 800):
    # Airfoil Boundary --> denotes where the airfoil is so that the streamlines know where to curve about
    # NumLines --> How many streamlines to show
    # ylow and yhigh --> vertical dimensions of the field
    # xstart --> horizontal starting point for the streamlines
    # tinteg --> how long to integrate for. calculated as chord length/flow speed
    # numpoints --> how many points to integrate to create a streamline

    startpos =  np.linspace(ylow, yhigh, NumLines) #creates starting nodes for NumLines number of streamlines
    lines = []

    for index in startpos:

        def integrand(t, pos):
            u, v = ArbitraryV(pos[0], pos[1])
            return [u,v]
        # provides velocities at each of the 800 points i've created. will be integrated later to derive position of streamline

        def Flag(t, pos):
            if AirfoilBoundary.contains_point(pos):
                return -1.0
            else:
                return 1.0
        # checks if the streamline is entering the airfoil's body

        Flag.terminal = True    
        Flag.direction = -1
        # instruction set to be used later. 
        # if the condition of flag is met, terminal is true and we must stop.
        # if the return value flips, direction is -1 and we must stop.

        solution = solve_ivp(integrand, [0.0, tinteg], [xstart, index], max_step = 0.008, t_eval = np.linspace(0.0, tinteg, numpoints), events = Flag, dense_output= False)

        xsolution, ysolution = solution.y
        # breaks the formed 2-D solution array into two 1-D arrays with x and y positions
        OutsideFlagList = []
        for x,y in zip(xsolution, ysolution):
            result = not AirfoilBoundary.contains_point((x,y))
            OutsideFlagList.append(result)

        OutsideFlag = np.array(OutsideFlagList)

        # Now we combine all the points that are outside the airfoil into lines 
        segx, segy = [], []
        for flag, x, y in zip(OutsideFlag, xsolution, ysolution):
            if flag:
                segx.append(x)
                segy.append(y)
            else:
                if len(segx) > 3:
                    lines.append((np.array(segx), np.array(segy)))
                segx, segy = [], []

        if len(segx) > 3:
            lines.append((np.array(segx), np.array(segy)))

    return lines

Darkbg = "#0b0d10"

def PlottingFunction(ax, lines, xupper, yupper, xlower, ylower, mycode, aoa):
    ax.set_facecolor(Darkbg)
    ax.tick_params(colors = "#aaa")
    for spines in ax.spines.values():
        spines.set_edgecolor("#444")

    for xsolution, ysolution in lines:
        ax.plot(xsolution, ysolution, color = "#56c8f5", solid_capstyle = "round")

    plotx = np.concatenate([xupper, xlower[::-1], [xupper[0]]])
    ploty = np.concatenate([yupper, ylower[::-1], [yupper[0]]])
    ax.fill(plotx, ploty, color = "#cdd1d8", zorder = 5)
    ax.plot(plotx, ploty, color = "#ffffff", zorder = 5)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-0.7, 0.7)
    ax.set_aspect("equal")
    ax.set_title(f"NACA {mycode} ; {aoa}°", fontsize = 12)
    ax.set_xlabel("x / c", color="#aaa")
    ax.set_ylabel("y / c", color="#aaa")

# Main

print("Virtual Wind Tunnel - Visualising Streamlines")

print("Input 2412 if you're not aware of many codes")
raw_code = input("NACA Code: ")
raw_code = raw_code.strip()

print("Input 5 if you're not sure of other values")
raw_aoa = input("Angle of Attack: ")
raw_aoa = raw_aoa.strip()

try:
    aoa = float(raw_aoa)
except ValueError:
    aoa = 5.0

print("Running Python Code....")

xupper, yupper, xlower, ylower = CreateAirfoil(raw_code, n=120)
xupper = np.array(xupper)
yupper = np.array(yupper)
xlower = np.array(xlower)
ylower = np.array(ylower)
xpanel, ypanel = CreatePanels(xupper, yupper, xlower, ylower)
ArbitraryV = SolveEquations(xpanel, ypanel, aoa)

xarray = np.concatenate([xupper, xlower[::-1]])
yarray = np.concatenate([yupper, ylower[::-1]])
AirfoilBoundary = mpath.Path(np.column_stack([xarray, yarray]))

lines = CreateStreamlines(ArbitraryV, AirfoilBoundary)

figure, ax = plt.subplots(figsize = (11, 5))
figure.patch.set_facecolor(Darkbg)
PlottingFunction(ax, lines, xupper, yupper, xlower, ylower, raw_code, aoa)
figure.tight_layout(pad = 1.2)

plt.show()


