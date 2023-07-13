import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib.colors import LinearSegmentedColormap

from typing import List
from typing import Optional


class Data(object):
    tl=pd.read_csv("../data/testData/AugmentedTrainingLogTestData.csv")
    et=pd.read_csv("../data/testData/ExerciseTypeTestData.csv")
    e=pd.read_csv("../data/testData/ExerciseTestData.csv")
    ms=pd.read_csv("../data/generatedData/Client1.ms.csv")
    SquatID=e.query("Name==\"Squat\"")["Id"].iloc[0]
    BenchID=e.query("Name==\"Bench\"")["Id"].iloc[0]
    DeadliftID=e.query("Name==\"Deadlift\"")["Id"].iloc[0]


IMG_OUT_DIR="images"
TITLE_FONT_SIZE=12
SUBTITLE_FONT_SIZE=10
UNITS={
    "Volume": "lbs",
    "Effort": "RPE",
    "Intensity": "% of 1RM",
    "Sets": "Count",
    "Reps": "Count",
}


def listToFilter(_in: List[str]) -> str:
    rv=""
    for i,item in enumerate(_in):
        rv+=("'{0}'".format(item))
        if i+1<len(_in):
            rv+=","
    return rv


def getUnit(_in: str) -> str:
    return "({0})".format(UNITS[_in]) if _in in UNITS else ""


class Ch3(object):
    fileLoc="{0}/ch3".format(IMG_OUT_DIR)
    __mergedDf=pd.DataFrame()

    @property
    def mergedDf(self):
        if Ch3.__mergedDf.empty:
            tmp=pd.merge(Data.tl,Data.e,left_on="ExerciseID", right_on="Id", how="outer")
            Ch3.__mergedDf=pd.merge(
                tmp,Data.et,left_on="TypeID", right_on="Id", how="outer",
            )
        return Ch3.__mergedDf

    @staticmethod
    def oneVTwo(one: str,
        two: str,
        trendline: Optional[bool]=True,
        types: Optional[List[str]]=["Main Compound","Main Compound Accessory"]
    ) -> None:
        tmp=Ch3().mergedDf.query(
            "Intensity<=1.2 and (T in ({0}))".format(listToFilter(types)),
        )[[one,two]].dropna()

        if trendline:
            model=np.poly1d(np.polyfit(tmp[one],tmp[two],3))
            line=np.arange(0,max(tmp[one]),max(tmp[one])/1000)

        fig,ax=plt.subplots(figsize=(12,6))
        ax.scatter(tmp[one],tmp[two],s=20,alpha=0.5)
        if trendline:
            ax.plot(line,model(line))

        plt.ylim(0,max(tmp[two]))
        plt.title(
            "{0} vs {1} for Main Compound and Main Compound Accessories".format(
                one,two,
        ), fontsize=TITLE_FONT_SIZE)
        plt.xlabel("{0} {1}".format(one,getUnit(one)))
        plt.ylabel("{0} {1}".format(two,getUnit(two)))
        plt.grid()
        plt.savefig("{0}/{1}Vs{2}.png".format(Ch3.fileLoc,one,two),
            bbox_inches="tight"
        )
        fig.clf()
        plt.clf()
        plt.close('all')
        plt.close(fig)
        ax.cla()

    @staticmethod
    def intensityVsVolume() -> None:
        types: List[str]=["Main Compound","Main Compound Accessory"]
        one: str="Intensity"
        two: str="Volume"
        fig,ax=plt.subplots(figsize=(12,6))
        maxY: int=0
        for i in range(5,11):
            tmp=Ch3().mergedDf.query(
                "Effort=={0} and Intensity<=1.2 and (T in ({1}))".format(i,listToFilter(types)),
            )[[one,two]].dropna()

            if len(tmp)>0:
                model=np.poly1d(np.polyfit(tmp[one],tmp[two],2))
                line=np.arange(0,max(tmp[one]),max(tmp[one])/1000)

                ax.scatter(tmp[one],tmp[two],s=20,alpha=0.5,color=((i-5)/5,0,0,0))
                ax.plot(line,model(line),color=((i-5)/5,0,0,1))
                if max(tmp[two])>maxY:
                    maxY=max(tmp[two])

        plt.ylim(0,maxY)
        plt.title(
            "{0} vs {1} for Main Compound and Main Compound Accessories".format(
                one,two,
        ), fontsize=TITLE_FONT_SIZE)
        plt.xlabel("{0} {1}".format(one,getUnit(one)))
        plt.ylabel("{0} {1}".format(two,getUnit(two)))
        plt.grid()
        plt.savefig("{0}/intensityVsVolumeSplitRPE.png".format(Ch3.fileLoc,one,two),
            bbox_inches="tight"
        )
        fig.clf()
        plt.clf()
        plt.close('all')
        plt.close(fig)
        ax.cla()

    @staticmethod
    def basicPotentialSurface(
        liftName: str,
        liftId: int,
        date: str,
        effort: int,
        interWorkoutFatigue: int=0,
        interExerciseFatigue: int=0,
    ) -> None:
        msVals=Data.ms.query(
            "ClientID==1 and StateGeneratorID==1 and PotentialSurfaceID==1 and ExerciseID=={0} and Date==\"{1}\""
            .format(liftId,date)
        )
        X=np.arange(1,15,0.01)
        Y=np.arange(1,15,0.01)
        X,Y=np.meshgrid(X,Y)
        Z=(msVals["Eps"].iloc[0]+
           msVals["Eps1"].iloc[0]*effort-
           msVals["Eps2"].iloc[0]*interWorkoutFatigue-
           msVals["Eps3"].iloc[0]*interExerciseFatigue-
           msVals["Eps4"].iloc[0]*((X-1)**2)*((Y-1)**2)-
           msVals["Eps5"].iloc[0]*((X-1)**2)-
           msVals["Eps6"].iloc[0]*((Y-1)**2)
        )
        Z[Z<0]=np.nan
        maxIntensity=Z[0][0]

        fig=plt.figure(figsize=(12,6))
        ax=fig.add_subplot(111,projection='3d')
        ax.view_init(40,-30)
        surf=ax.plot_surface(X,Y,Z,
            cmap=LinearSegmentedColormap.from_list("",[(0,0,0,0.5),"red"]),
            edgecolor="black",
            linewidth=0.1,
            antialiased=True
        )
        ax.set_zlim(0,1.05)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.set_zlabel("{0} {1}".format("Intensity",getUnit("Intensity")))
        fig.colorbar(surf,shrink=0.7,aspect=15,pad=0.09)
        plt.suptitle(
            "Potential Surface for {0} At RPE {1}"
            .format(liftName,effort),
            fontsize=TITLE_FONT_SIZE,
            fontweight="bold",
            x=(fig.subplotpars.right+fig.subplotpars.left)/2,
            horizontalalignment="center",
        )
        plt.title(
                "I(1,1)={0} MSE: {1:0.2E}\nDate: {2}  Window: {3} days  Time frame: {4} days".format(
                maxIntensity,
                msVals["Mse"].iloc[0],
                date,
                msVals["Win"].iloc[0],
                msVals["TimeFrame"].iloc[0],
            ), fontsize=SUBTITLE_FONT_SIZE,
        )
        plt.xlim(1,15)
        plt.ylim(1,15)
        plt.xlabel("{0} {1}".format("Sets",getUnit("Sets")))
        plt.ylabel("{0} {1}".format("Reps",getUnit("Reps")))
        plt.savefig("{0}/PotentialSurface/{1}Effort{2}.basic.png"
            .format(Ch3.fileLoc,liftName,effort),
            bbox_inches="tight"
        )
        fig.clf()
        plt.clf()
        plt.close('all')
        plt.close(fig)
        ax.cla()

    @staticmethod
    def volumeBasePotentialSurface(
        liftName: str,
        liftId: int,
        date: str,
        effort: int,
        interWorkoutFatigue: int=0,
        interExerciseFatigue: int=0,
    ) -> None:
        msVals=Data.ms.query(
            "ClientID==1 and StateGeneratorID==1 and PotentialSurfaceID==2 and ExerciseID=={0} and Date==\"{1}\""
            .format(liftId,date)
        )
        X=np.arange(1,15,0.01)
        Y=np.arange(1,15,0.01)
        X,Y=np.meshgrid(X,Y)
        # Z=np.sqrt((msVals["Eps1"].iloc[0]*effort)/(1+
        #     msVals["Eps2"].iloc[0]*interWorkoutFatigue+
        #     msVals["Eps3"].iloc[0]*interExerciseFatigue+
        #     msVals["Eps4"].iloc[0]*((X-1)**2)*((Y-1)**2)+
        #     msVals["Eps5"].iloc[0]*((X-1)**2)+
        #     msVals["Eps6"].iloc[0]*((Y-1)**2)+
        #     msVals["Eps"].iloc[0]*msVals["Eps1"].iloc[0]*effort
        # ))
        Z=np.sqrt(1/(
            msVals["Eps"].iloc[0]+
            msVals["Eps1"].iloc[0]*1/effort+
            msVals["Eps2"].iloc[0]*interWorkoutFatigue/effort+
            msVals["Eps3"].iloc[0]*interExerciseFatigue/effort+
            msVals["Eps4"].iloc[0]*((X-1)**2)*((Y-1)**2)/effort+
            msVals["Eps5"].iloc[0]*((X-1)**2)/effort+
            msVals["Eps6"].iloc[0]*((Y-1)**2)/effort
        ))
        Z[Z<0]=np.nan
        maxIntensity=Z[0][0]

        fig=plt.figure(figsize=(12,6))
        ax=fig.add_subplot(111,projection='3d')
        ax.view_init(40,-30)
        surf=ax.plot_surface(X,Y,Z,
            cmap=LinearSegmentedColormap.from_list("",[(0,0,0,0.5),"red"]),
            edgecolor="black",
            linewidth=0.1,
            antialiased=True
        )
        ax.set_zlim(0,1.05)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.set_zlabel("{0} {1}".format("Intensity",getUnit("Intensity")))
        fig.colorbar(surf,shrink=0.7,aspect=15,pad=0.09)
        plt.suptitle(
            "Potential Surface for {0} At RPE {1}"
            .format(liftName,effort),
            fontsize=TITLE_FONT_SIZE,
            fontweight="bold",
            x=(fig.subplotpars.right+fig.subplotpars.left)/2,
            horizontalalignment="center",
        )
        plt.title(
                "I(1,1)={0} MSE: {1:0.2E}\nDate: {2}  Window: {3} days  Time frame: {4} days".format(
                maxIntensity,
                msVals["Mse"].iloc[0],
                date,
                msVals["Win"].iloc[0],
                msVals["TimeFrame"].iloc[0],
            ), fontsize=SUBTITLE_FONT_SIZE,
        )
        plt.xlim(1,15)
        plt.ylim(1,15)
        plt.xlabel("{0} {1}".format("Sets",getUnit("Sets")))
        plt.ylabel("{0} {1}".format("Reps",getUnit("Reps")))
        plt.savefig("{0}/PotentialSurface/{1}Effort{2}.volumeBase.png"
            .format(Ch3.fileLoc,liftName,effort),
            bbox_inches="tight"
        )
        fig.clf()
        plt.clf()
        plt.close('all')
        plt.close(fig)
        ax.cla()

    @staticmethod
    def volumeFromBasicPotentialSurface(
        liftName: str,
        liftId: int,
        date: str,
        effort: int,
        interWorkoutFatigue: int=0,
        interExerciseFatigue: int=0,
    ) -> None:
        msVals=Data.ms.query(
            "ClientID==1 and StateGeneratorID==1 and ExerciseID=={0} and Date==\"{1}\""
            .format(liftId,date)
        )
        X=np.arange(1,25,0.1)
        Y=np.arange(1,25,0.1)
        X,Y=np.meshgrid(X,Y)
        Z=(msVals["Eps"].iloc[0]+
           msVals["Eps1"].iloc[0]*effort-
           msVals["Eps2"].iloc[0]*0-    # This part is left out of chapter 3
           msVals["Eps3"].iloc[0]*interWorkoutFatigue-
           msVals["Eps4"].iloc[0]*interExerciseFatigue-
           msVals["Eps5"].iloc[0]*((X-1)**2)*((Y-1)**2)-
           msVals["Eps6"].iloc[0]*((X-1)**2)-
           msVals["Eps7"].iloc[0]*((Y-1)**2)
        )
        Z[Z<0]=np.nan
        V=X*Y*Z
        # maxIntensity=Z[0][0]

        fig=plt.figure(figsize=(12,6))
        ax=fig.add_subplot(111,projection='3d')
        ax.view_init(40,-30)
        surf=ax.plot_surface(X,Y,V,
            cmap=LinearSegmentedColormap.from_list("",[(0,0,0,1),(0.9,0.9,0.9,1)]),
            edgecolor="black",
            linewidth=0,
            antialiased=False
        )
        # ax.set_zlim(0,1.05)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.set_zlabel("{0} {1}".format("Volume from Intensity","(% Volume)"))
        fig.colorbar(surf,shrink=0.7,aspect=15,pad=0.09)
        plt.suptitle(
            "Volume From Potential Surface for {0} At RPE {1}"
            .format(liftName,effort),
            fontsize=TITLE_FONT_SIZE,
            fontweight="bold",
            x=(fig.subplotpars.right+fig.subplotpars.left)/2,
            horizontalalignment="center",
        )
        # plt.title(
        #     "I(1,1)={0}".format(maxIntensity),
        #     fontsize=SUBTITLE_FONT_SIZE,
        # )
        plt.xlim(1,25)
        plt.ylim(1,25)
        plt.xlabel("{0} {1}".format("Sets",getUnit("Sets")))
        plt.ylabel("{0} {1}".format("Reps",getUnit("Reps")))
        plt.savefig("{0}/Volume/{1}Effort{2}.png"
            .format(Ch3.fileLoc,liftName,effort),
            bbox_inches="tight"
        )
        fig.clf()
        plt.clf()
        plt.close('all')
        plt.close(fig)
        ax.cla()


if __name__=="__main__":
    # Ch3.intensityVsVolume()
    print("Making intensity vs volume scatterplot")
    Ch3.oneVTwo("Intensity","Volume")
    print("Making effort vs volume scatterplot")
    Ch3.oneVTwo("Effort","Volume",False)
    print("Making intensity vs effort scatterplot")
    Ch3.oneVTwo("Intensity","Effort")
    print("Making sets vs reps scatterplot")
    Ch3.oneVTwo("Sets","Reps",False)
    for i in range(5,11):
        for n,e in zip(
            ["Squat","Bench","Deadlift"],
            [Data.SquatID,Data.BenchID,Data.DeadliftID],
        ):
            print("Making basic potential surface for {0} at RPE {1}".format(n,i))
            Ch3.basicPotentialSurface(liftName=n,
                liftId=e,
                date="05/06/2023",
                effort=i,
            )
            print("Making volume base potential surface for {0} at RPE {1}".format(n,i))
            Ch3.volumeBasePotentialSurface(liftName=n,
                liftId=e,
                date="05/06/2023",
                effort=i,
            )
            print("Making volume from potential surface for {0} at RPE {1}".format(n,i))
            Ch3.volumeFromBasicPotentialSurface(liftName=n,
                liftId=e,
                date="05/06/2023",
                effort=i,
            )
