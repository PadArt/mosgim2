import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import numpy as np
import datetime


def plot2l(out_file, colat, lon, ts, frames1, frames2):

    time = np.array([datetime.datetime.utcfromtimestamp(float(t)) for t in ts])

    lon_m, colat_m = np.meshgrid(lon, colat)

    def some_data(i):   # function returns a 2D data array
        return frames1[i], frames2[i] 

    fig = plt.figure(figsize=(5, 7))

    ax1 = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(3, 1, 2, projection=ccrs.PlateCarree())
    ax3 = fig.add_subplot(3, 1, 3, projection=ccrs.PlateCarree())

    m1 = np.max(np.array(frames1))    
    m2 = np.max(np.array(frames2))    
    m3 = np.max(np.array(frames1) + np.array(frames1))    

    levels1=np.arange(-0.5,m1,0.5)
    levels2=np.arange(-0.5,m2,0.5)
    levels3=np.arange(-0.5,m3,0.5)

    cont1 = ax1.contourf(lon_m, 90.-colat_m, some_data(0)[0], levels1,  cmap=plt.cm.jet, transform=ccrs.PlateCarree())    # first image on screen
    cont2 = ax2.contourf(lon_m, 90.-colat_m, some_data(0)[1], levels2,  cmap=plt.cm.jet, transform=ccrs.PlateCarree())    # first image on screen
    cont3 = ax3.contourf(lon_m, 90.-colat_m, some_data(0)[0] + some_data(0)[1], levels3,  cmap=plt.cm.jet, transform=ccrs.PlateCarree())    # first image on screen

    ax1.set_title('layer1, '+ str(time[0]))  
    ax2.set_title('layer2, '+ str(time[0]))  
    ax3.set_title('GIM, '+ str(time[0]))  

    ax1.coastlines()
    ax2.coastlines()
    ax3.coastlines()

    fig.colorbar(cont1, ax=ax1)
    fig.colorbar(cont2, ax=ax2)
    fig.colorbar(cont3, ax=ax3)
    plt.tight_layout()

    # animation function
    def animate(i):
        global cont1, cont2, cont3
        z1, z2 = some_data(i)
        cont1 = ax1.contourf(lon_m, 90.-colat_m, z1, levels1,  cmap=plt.cm.jet, transform=ccrs.PlateCarree())
        cont2 = ax2.contourf(lon_m, 90.-colat_m, z2, levels2,  cmap=plt.cm.jet, transform=ccrs.PlateCarree())
        cont3 = ax3.contourf(lon_m, 90.-colat_m, z1+z2, levels3,  cmap=plt.cm.jet, transform=ccrs.PlateCarree())
        ax1.set_title('layer1, '+ str(time[i]))  
        ax2.set_title('layer2, '+ str(time[i]))  
        ax3.set_title('GIM, '+ str(time[i]))  
        return cont1, cont2, cont3

    anim = animation.FuncAnimation(fig, animate, frames=len(ts), repeat=False)
    anim.save(out_file, writer='imagemagick')

def plot1l(out_file, colat, lon, ts, frames1):

    time = np.array([datetime.datetime.utcfromtimestamp(float(t)) for t in ts])

    lon_m, colat_m = np.meshgrid(lon, colat)

    def some_data(i):   # function returns a 2D data array
        return frames1[i] 

    fig = plt.figure(figsize=(4.5, 3))

    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    m1 = np.max(np.array(frames1))    

    levels1=np.arange(-0.5,m1,0.5)

    cont1 = ax1.contourf(lon_m, 90.-colat_m, some_data(0), levels1,  cmap=plt.cm.jet, transform=ccrs.PlateCarree())    # first image on screen

    ax1.set_title('GIM, '+ str(time[0]))  

    ax1.coastlines()

    fig.colorbar(cont1, ax=ax1, orientation='horizontal', pad=0.05)
    plt.tight_layout()

    # animation function
    def animate(i):
        global cont1
        z1 = some_data(i)
        cont1 = ax1.contourf(lon_m, 90.-colat_m, z1, levels1,  cmap=plt.cm.jet, transform=ccrs.PlateCarree())
        ax1.set_title('GIM, '+ str(time[i]))  
        return cont1

    anim = animation.FuncAnimation(fig, animate, frames=len(ts), repeat=False)
    anim.save(out_file, writer='imagemagick')
