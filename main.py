import glob
import numpy as np
import pandas as pd
from time import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from typing import Any
from typing import List
from typing import Dict
from typing import ClassVar


IMG_OUT_DIR="images"
TITLE_FONT_SIZE=12
SUBTITLE_FONT_SIZE=10
IMAGE_SIZES: Dict[str,Dict[str,int]]={
    "fullPage": {
        "x": 800,
        "y": 1100,
    }, "halfPage": {
        "x": 800,
        "y": 600,
    }, "thirdPage": {
        "x": 800,
        "y": 500,
    }, "square": {
        "x": 700,
        "y": 600,
    },
}


timerIndent: int=-4
def timerDecorator(f: callable) -> callable:
    def rv(*arg, **kwargs) -> Any:
        global timerIndent
        timerIndent+=4
        print("{0}Starting {1} with args {2} and kwargs {3}..."
              .format(" "*timerIndent,f.__name__,arg,kwargs)
        )
        start: int=time()
        rvs: Any=f(*arg,**kwargs)
        print("{0}Finished {1}: {2}"
              .format(" "*timerIndent,f.__name__,time()-start)
        )
        timerIndent-=4
        return rvs
    return rv


class Units(object):
    UNITS={
        "Volume": "lbs",
        "PercentVolume": "% lbs",
        "Effort": "RPE",
        "Intensity": "% of 1RM",
        "Sets": "Count",
        "Reps": "Count",
        "Date": "Time",
        "Constant": "Unitless",
    }

    @staticmethod
    def formatUnits(_in: List[str]) -> str:
        rv="("
        for i,u in enumerate(_in):
            rv+=("{0}," if i+1<len(_in) else "{0}").format(
                Units.UNITS[u]
            )
        rv+=")"
        return rv


class Data(object):
    tl=pd.read_csv("../data/testData/AugmentedTrainingLogTestData.csv")
    et=pd.read_csv("../data/testData/ExerciseTypeTestData.csv")
    e=pd.read_csv("../data/testData/ExerciseTestData.csv")
    ms=pd.read_csv("../data/generatedData/Client1.ms.csv")
    vs=pd.read_csv("../data/generatedData/Client1.volSkew.csv")
    approxVolSkew={
        f[f.rfind('/')+1:f.rfind('.')]: pd.read_csv(f)
        for f in glob.glob("../data/generatedData/basicSurfVolSkewApprox/*.csv")
    }
    SquatID=e.query("Name==\"Squat\"")["Id"].iloc[0]
    BenchID=e.query("Name==\"Bench\"")["Id"].iloc[0]
    DeadliftID=e.query("Name==\"Deadlift\"")["Id"].iloc[0]
    liftNameToId: Dict[str,int]={
        "Squat": SquatID,
        "Bench": BenchID,
        "Deadlift": DeadliftID,
    }

    @staticmethod
    def listToFilter(_in: List[str]) -> str:
        rv=""
        for i,item in enumerate(_in):
            rv+=("'{0}'".format(item))
            if i+1<len(_in):
                rv+=","
        return rv


class Ch3(object):
    fileLoc: ClassVar[str]="{0}/ch3".format(IMG_OUT_DIR)
    __mergedDf=pd.DataFrame()
    if __mergedDf.empty:
        tmp=pd.merge(Data.tl,Data.e,left_on="ExerciseID", right_on="Id", how="outer")
        __mergedDf=pd.merge(
            tmp,Data.et,left_on="TypeID", right_on="Id", how="outer",
        )

    @staticmethod
    @timerDecorator
    def allPlots(date: str) -> None:
        Ch3.intensityVsVolume()
        Ch3.effortVsVolume()
        Ch3.intensityVsEffort()
        Ch3.setsVsReps()
        Ch3.allPotentialSurfaces(date="05/06/2023")
        Ch3.allVolumeSurfaces(date="05/06/2023")
        Ch3.multiBasicSurface(effort=[5,10],date=date,liftName="Squat")
        Ch3.multiBasicSurfaceVolume(effort=[5,10],date=date,liftName="Squat")
        Ch3.eps1OverTime()
        Ch3.volumeSkewOverTime()
        Ch3.approxVolumeSkewGraphs()

    @staticmethod
    @timerDecorator
    def intensityVsVolume() -> None:
        tmp=Ch3.__mergedDf.query(
            "Intensity<=1.2 and (T in ({0}))".format(Data.listToFilter([
                "Main Compound",
                "Main Compound Accessory",
            ])),
        )[["Intensity","Volume","Effort"]].dropna()
        fig=px.scatter(tmp,x="Intensity",y="Volume",color="Effort",
            labels={
                "Intensity": "Intensity {0}".format(Units.formatUnits(["Intensity"])),
                "Volume": "Volume {0}".format(Units.formatUnits(["Volume"])),
            }, 
            title="Intensity Vs Volume for Main Compound and Main Compound Accessories",
        )
        fig.update_layout(
            title={
                "x": 0.5,
                "xanchor": "center",
            },
            width=IMAGE_SIZES["thirdPage"]["x"],
            height=IMAGE_SIZES["thirdPage"]["y"],
        )
        fig.write_image("{0}/intensityVsVolume.png".format(Ch3.fileLoc))

    @staticmethod
    @timerDecorator
    def effortVsVolume() -> None:
        tmp=Ch3.__mergedDf.query(
            "Intensity<=1.2 and (T in ({0}))".format(Data.listToFilter([
                "Main Compound",
                "Main Compound Accessory",
            ])),
        )[["Intensity","Volume","Effort"]].dropna()
        fig=px.scatter(tmp,x="Effort",y="Volume",color="Intensity",
            labels={
                "Effort": "Effort {0}".format(Units.formatUnits(["Effort"])),
                "Volume": "Volume {0}".format(Units.formatUnits(["Volume"])),
            }, 
            title="Effort Vs Volume for Main Compound and Main Compound Accessories",
        )
        fig.update_layout(
            title={
                "x": 0.5,
                "xanchor": "center",
            },
            width=IMAGE_SIZES["thirdPage"]["x"],
            height=IMAGE_SIZES["thirdPage"]["y"],
        )
        fig.write_image("{0}/effortVsVolume.png".format(Ch3.fileLoc))

    @staticmethod
    @timerDecorator
    def intensityVsEffort() -> None:
        tmp=Ch3.__mergedDf.query(
            "Intensity<=1.2 and (T in ({0}))".format(Data.listToFilter([
                "Main Compound",
                "Main Compound Accessory",
            ])),
        )[["Intensity","Volume","Effort"]].dropna()
        fig=px.scatter(tmp,x="Intensity",y="Effort",color="Volume",
            labels={
                "Intensity": "Intensity {0}".format(Units.formatUnits(["Intensity"])),
                "Effort": "Effort {0}".format(Units.formatUnits(["Effort"])),
            }, 
            title="Intensity Vs Effort for Main Compound and Main Compound Accessories",
        )
        fig.update_layout(
            title={
                "x": 0.5,
                "xanchor": "center",
            },
            width=IMAGE_SIZES["thirdPage"]["x"],
            height=IMAGE_SIZES["thirdPage"]["y"],
        )
        fig.write_image("{0}/intensityVsEffort.png".format(Ch3.fileLoc))

    @staticmethod
    @timerDecorator
    def setsVsReps() -> None:
        tmp=Ch3.__mergedDf.query(
            "Intensity<=1.2 and (T in ({0}))".format(Data.listToFilter([
                "Main Compound",
                "Main Compound Accessory",
            ])),
        )[["Sets","Reps","Volume"]].dropna()
        fig=px.scatter(tmp,x="Sets",y="Reps",color="Volume",
            labels={
                "Sets": "Sets {0}".format(Units.formatUnits(["Sets"])),
                "Reps": "Reps {0}".format(Units.formatUnits(["Reps"])),
            },
            title="Intensity Vs Effort for Main Compound and Main Compound Accessories",
        )
        # invX=np.arange(1,tmp["Sets"].max(),step=0.1)
        # for i in range(5,26,5):
        #     invY=i/invX
        #     fig.add_trace(go.Scatter(x=invX,y=invY,mode="lines"))
        # fig.add_trace(go.Scatter(x=invX,y=invX,mode="lines"))
        fig.update_layout(
            title={
                "x": 0.5,
                "xanchor": "center",
            },
            width=IMAGE_SIZES["thirdPage"]["x"],
            height=IMAGE_SIZES["thirdPage"]["y"],
        )
        fig.write_image("{0}/setsVsReps.png".format(Ch3.fileLoc))

    @staticmethod
    @timerDecorator
    def allSurfaces(date: str) -> None:
        Ch3.allPotentialSurfaces(date)
        Ch3.allVolumeSurfaces(date)

    @staticmethod
    @timerDecorator
    def allPotentialSurfaces(date: str) -> None:
        for e in range(4,11):
            for l in Data.liftNameToId:
                Ch3.basicSurface(effort=e,date=date,liftName=l)
                Ch3.volumeBaseSurface(effort=e,date=date,liftName=l)

    @staticmethod
    @timerDecorator
    def basicSurface(effort: int, date: str, liftName: str) -> None:
        interWorkoutFatigue: int=0
        interExerciseFatigue: int=0
        msVals=Data.ms.query(
            "ClientID==1 and StateGeneratorID==1 and PotentialSurfaceID==1 and ExerciseID=={0} and Date==\"{1}\""
            .format(Data.liftNameToId[liftName],date)
        )
        X=np.arange(1,15,0.1)
        Y=np.arange(1,15,0.1)
        X,Y=np.meshgrid(X,Y)
        Z=(msVals["Eps"].iloc[0]+
           msVals["Eps1"].iloc[0]*effort-
           msVals["Eps2"].iloc[0]*interWorkoutFatigue-
           msVals["Eps3"].iloc[0]*interExerciseFatigue-
           msVals["Eps4"].iloc[0]*((X-1)**2)*((Y-1)**2)-
           msVals["Eps5"].iloc[0]*((X-1)**2)-
           msVals["Eps6"].iloc[0]*((Y-1)**2)
        )
        Z[Z<-0.1]=np.nan
        maxIntensity=Z[0][0]
        fig=go.Figure(data=[go.Surface(x=X, y=Y, z=Z,
            contours={
                "x": {"show": True, "start": 0, "end": 15},
                "y": {"show": True, "start": 0, "end": 15},
            },
        )])
        fig.update_layout(
            scene={
                "xaxis_title": "Sets {0}".format(Units.formatUnits(["Sets"])),
                "yaxis_title": "Reps {0}".format(Units.formatUnits(["Reps"])),
                "zaxis_title": "Intensity {0}".format(Units.formatUnits(["Intensity"])),
                "zaxis": { "nticks": 12, "range": [0,1.3], },
            },
            title={
                "text": "Basic Potential Surface for {0} at RPE {1}<br>"
                        "<sup>I(0,0)={2}  MSE: {3:0.2E}<br>"
                        "Date: {4}  Window: {5}  Time Frame: {6}</sup>"
                        .format(
                            liftName,
                            effort,
                            maxIntensity,
                            msVals["Mse"].iloc[0],
                            date,
                            msVals["Win"].iloc[0],
                            msVals["TimeFrame"].iloc[0],
                        ),
                "x": 0.5,
                "xanchor": "center",
            },
            width=IMAGE_SIZES["square"]["x"],
            height=IMAGE_SIZES["square"]["y"],
            margin={
                "b": 30,
            },
        )
        fig.write_image(
            "{0}/PotentialSurface/{1}.Effort{2}.basic.png"
            .format(Ch3.fileLoc,liftName,effort)
        )

    #TODO -make a version of this for volume base surface
    @staticmethod
    @timerDecorator
    def multiBasicSurface(effort: List[int], date: str, liftName: str) -> None:
        interWorkoutFatigue: int=0
        interExerciseFatigue: int=0
        msVals=Data.ms.query(
            "ClientID==1 and StateGeneratorID==1 and PotentialSurfaceID==1 and ExerciseID=={0} and Date==\"{1}\""
            .format(Data.liftNameToId[liftName],date)
        )
        fig=make_subplots(cols=1,rows=len(effort),specs=[
                [{"type": "surface"}] for i in range(0,len(effort))
            ], 
            subplot_titles=tuple([
                "Basic Potential Surface for {0} at RPE {1}<br>"
                "<sup>I(0,0)={2}  MSE: {3:0.2E}<br>"
                "Date: {4}  Window: {5}  Time Frame: {6}</sup>"
                .format(
                    liftName,
                    e,
                    (msVals["Eps"].iloc[0]+
                       msVals["Eps1"].iloc[0]*e-
                       msVals["Eps2"].iloc[0]*interWorkoutFatigue-
                       msVals["Eps3"].iloc[0]*interExerciseFatigue
                    ),
                    msVals["Mse"].iloc[0],
                    date,
                    msVals["Win"].iloc[0],
                    msVals["TimeFrame"].iloc[0],
                ) for e in effort
            ]),
            vertical_spacing=0.1,
        )
        for i,e in enumerate(effort):
            X=np.arange(1,15,0.1)
            Y=np.arange(1,15,0.1)
            X,Y=np.meshgrid(X,Y)
            Z=(msVals["Eps"].iloc[0]+
               msVals["Eps1"].iloc[0]*e-
               msVals["Eps2"].iloc[0]*interWorkoutFatigue-
               msVals["Eps3"].iloc[0]*interExerciseFatigue-
               msVals["Eps4"].iloc[0]*((X-1)**2)*((Y-1)**2)-
               msVals["Eps5"].iloc[0]*((X-1)**2)-
               msVals["Eps6"].iloc[0]*((Y-1)**2)
            )
            Z[Z<-0.1]=np.nan
            maxIntensity=Z[0][0]
            fig.add_trace(go.Surface(x=X, y=Y, z=Z,
                contours={
                    "x": {"show": True, "start": 0, "end": 15},
                    "y": {"show": True, "start": 0, "end": 15},
                },
                colorbar={
                    "len": 1/len(effort),
                    "y": i/len(effort)+1/(2*len(effort)),
                },
            ), row=i+1, col=1)
            fig.update_scenes({
                    "xaxis_title": "Sets {0}".format(Units.formatUnits(["Sets"])),
                    "yaxis_title": "Reps {0}".format(Units.formatUnits(["Reps"])),
                    "zaxis_title": "Intensity {0}".format(Units.formatUnits(["Intensity"])),
                    "zaxis": { "nticks": 12, "range": [0,1.3], },
                }, row=i+1, col=1,
            )
        fig.update_layout(
            width=IMAGE_SIZES["fullPage"]["x"],
            height=IMAGE_SIZES["fullPage"]["y"],
            margin={
                "b": 20,
            }, showlegend=False,
        )
        fig.write_image(
            "{0}/PotentialSurface/Dual{1}.Effort{2}.basic.png"
            .format(Ch3.fileLoc,liftName,str(effort).replace(" ",""))
        )

    @staticmethod
    @timerDecorator
    def volumeBaseSurface(effort: int, date: str, liftName: str) -> None:
        interWorkoutFatigue: int=0
        interExerciseFatigue: int=0
        msVals=Data.ms.query(
            "ClientID==1 and StateGeneratorID==1 and PotentialSurfaceID==2 and ExerciseID=={0} and Date==\"{1}\""
            .format(Data.liftNameToId[liftName],date)
        )
        X=np.arange(1,15,0.1)
        Y=np.arange(1,15,0.1)
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
        Z[Z<-0.1]=np.nan
        maxIntensity=Z[0][0]
        fig=go.Figure(data=[go.Surface(x=X, y=Y, z=Z,
            contours={
                "x": {"show": True, "start": 0, "end": 15},
                "y": {"show": True, "start": 0, "end": 15},
            },
        )])
        fig.update_layout(
            scene={
                "xaxis_title": "Sets {0}".format(Units.formatUnits(["Sets"])),
                "yaxis_title": "Reps {0}".format(Units.formatUnits(["Reps"])),
                "zaxis_title": "Intensity {0}".format(Units.formatUnits(["Intensity"])),
                "zaxis": { "nticks": 12, "range": [0,1.3], },
            },
            title={
                "text": "Volume Base Potential Surface for {0} at RPE {1}<br>"
                        "<sup>I(0,0)={2}  MSE: {3:0.2E}<br>"
                        "Date: {4}  Window: {5}  Time Frame: {6}</sup>"
                        .format(
                            liftName,
                            effort,
                            maxIntensity,
                            msVals["Mse"].iloc[0],
                            date,
                            msVals["Win"].iloc[0],
                            msVals["TimeFrame"].iloc[0],
                        ),
                "x": 0.5,
                "xanchor": "center",
            },
            width=IMAGE_SIZES["square"]["x"],
            height=IMAGE_SIZES["square"]["y"],
            margin={
                "b": 30,
            },
        )
        fig.write_image(
            "{0}/PotentialSurface/{1}.Effort{2}.volumeBase.png"
            .format(Ch3.fileLoc,liftName,effort)
        )

    @staticmethod
    @timerDecorator
    def allVolumeSurfaces(date: str) -> None:
        for e in range(4,11):
            for l in Data.liftNameToId:
                Ch3.basicSurfaceVolume(effort=e,date=date,liftName=l)
                Ch3.volumeBaseSurfaceVolume(effort=e,date=date,liftName=l)

    @staticmethod
    @timerDecorator
    def basicSurfaceVolume(effort: int, date: str, liftName: str) -> None:
        interWorkoutFatigue: int=0
        interExerciseFatigue: int=0
        msVals=Data.ms.query(
            "ClientID==1 and StateGeneratorID==1 and PotentialSurfaceID==1 and ExerciseID=={0} and Date==\"{1}\""
            .format(Data.liftNameToId[liftName],date)
        )
        X=np.arange(1,35,0.1)
        Y=np.arange(1,35,0.1)
        X,Y=np.meshgrid(X,Y)
        Z=(msVals["Eps"].iloc[0]+
           msVals["Eps1"].iloc[0]*effort-
           msVals["Eps2"].iloc[0]*interWorkoutFatigue-
           msVals["Eps3"].iloc[0]*interExerciseFatigue-
           msVals["Eps4"].iloc[0]*((X-1)**2)*((Y-1)**2)-
           msVals["Eps5"].iloc[0]*((X-1)**2)-
           msVals["Eps6"].iloc[0]*((Y-1)**2)
        )
        V=X*Y*Z
        V[V<-10]=np.nan
        maxIntensity=Z[0][0]
        fig=go.Figure(data=[go.Surface(x=X, y=Y, z=V,
            contours={
                "x": {"show": True, "start": 0, "end": 35},
                "y": {"show": True, "start": 0, "end": 35},
            },
        )])
        fig.update_layout(
            scene={
                "xaxis_title": "Sets {0}".format(Units.formatUnits(["Sets"])),
                "yaxis_title": "Reps {0}".format(Units.formatUnits(["Reps"])),
                "zaxis_title": "Volume {0}".format(Units.formatUnits(["PercentVolume"])),
                "zaxis": { "nticks": 25, "range": [0,50], },
            }, scene_camera={
                "up": { "x": 0, "y": 0, "z": 1 },
                "center": { "x": 0, "y": 0, "z": 0 },
                "eye": { "x": 1, "y": -1, "z": 2 },
            }, title={
                "text": "Volume From Basic Potential Surface for {0} at RPE {1}<br>"
                        "<sup>I(0,0)={2}  MSE: {3:0.2E}<br>"
                        "Date: {4}  Window: {5}  Time Frame: {6}</sup>"
                        .format(
                            liftName,
                            effort,
                            maxIntensity,
                            msVals["Mse"].iloc[0],
                            date,
                            msVals["Win"].iloc[0],
                            msVals["TimeFrame"].iloc[0],
                        ),
                "x": 0.5,
                "xanchor": "center",
            },
            width=IMAGE_SIZES["square"]["x"],
            height=IMAGE_SIZES["square"]["y"],
            margin={
                "b": 30,
            },
        )
        fig.write_image(
            "{0}/Volume/{1}.Effort{2}.basic.png"
            .format(Ch3.fileLoc,liftName,effort)
        )

    # TODO - make a version of this for volume base surface
    @staticmethod
    @timerDecorator
    def multiBasicSurfaceVolume(effort: List[int], date: str, liftName: str) -> None:
        interWorkoutFatigue: int=0
        interExerciseFatigue: int=0
        msVals=Data.ms.query(
            "ClientID==1 and StateGeneratorID==1 and PotentialSurfaceID==1 and ExerciseID=={0} and Date==\"{1}\""
            .format(Data.liftNameToId[liftName],date)
        )
        fig=make_subplots(cols=1,rows=len(effort),specs=[
                [{"type": "surface"}] for i in range(0,len(effort))
            ], 
            subplot_titles=tuple([
                "Basic Potential Surface for {0} at RPE {1}<br>"
                "<sup>I(0,0)={2}  MSE: {3:0.2E}<br>"
                "Date: {4}  Window: {5}  Time Frame: {6}</sup>"
                .format(
                    liftName,
                    e,
                    (msVals["Eps"].iloc[0]+
                       msVals["Eps1"].iloc[0]*e-
                       msVals["Eps2"].iloc[0]*interWorkoutFatigue-
                       msVals["Eps3"].iloc[0]*interExerciseFatigue
                    ),
                    msVals["Mse"].iloc[0],
                    date,
                    msVals["Win"].iloc[0],
                    msVals["TimeFrame"].iloc[0],
                ) for e in effort
            ]),
            vertical_spacing=0.1,
        )
        for i,e in enumerate(effort):
            X=np.arange(1,25,0.1)
            Y=np.arange(1,25,0.1)
            X,Y=np.meshgrid(X,Y)
            Z=(msVals["Eps"].iloc[0]+
               msVals["Eps1"].iloc[0]*e-
               msVals["Eps2"].iloc[0]*interWorkoutFatigue-
               msVals["Eps3"].iloc[0]*interExerciseFatigue-
               msVals["Eps4"].iloc[0]*((X-1)**2)*((Y-1)**2)-
               msVals["Eps5"].iloc[0]*((X-1)**2)-
               msVals["Eps6"].iloc[0]*((Y-1)**2)
            )
            V=X*Y*Z
            V[V<-10]=np.nan
            maxIntensity=Z[0][0]
            fig.add_trace(go.Surface(x=X, y=Y, z=V,
                contours={
                    "x": {"show": True, "start": 0, "end": 25},
                    "y": {"show": True, "start": 0, "end": 25},
                },
                colorbar={
                    "len": 1/len(effort),
                    "y": i/len(effort)+1/(2*len(effort)),
                },
            ), row=i+1, col=1)
            fig.update_scenes({
                    "xaxis_title": "Sets {0}".format(Units.formatUnits(["Sets"])),
                    "yaxis_title": "Reps {0}".format(Units.formatUnits(["Reps"])),
                    "zaxis_title": "Volume {0}".format(Units.formatUnits(["PercentVolume"])),
                    "zaxis": { "nticks": 20, "range": [0,60], },
                }, camera={
                    "up": { "x": 0, "y": 0, "z": 1 },
                    "center": { "x": 0, "y": 0, "z": 0 },
                    "eye": { "x": 1, "y": -1, "z": 1.5 },
                }, row=i+1, col=1,
            )
        fig.update_layout(
            width=IMAGE_SIZES["fullPage"]["x"],
            height=IMAGE_SIZES["fullPage"]["y"],
            margin={
                "b": 20,
            }, showlegend=False,
        )
        fig.write_image(
            "{0}/Volume/Dual{1}.Effort{2}.basic.png"
            .format(Ch3.fileLoc,liftName,str(effort).replace(" ",""))
        )

    @staticmethod
    @timerDecorator
    def volumeBaseSurfaceVolume(effort: int, date: str, liftName: str) -> None:
        interWorkoutFatigue: int=0
        interExerciseFatigue: int=0
        msVals=Data.ms.query(
            "ClientID==1 and StateGeneratorID==1 and PotentialSurfaceID==1 and ExerciseID=={0} and Date==\"{1}\""
            .format(Data.liftNameToId[liftName],date)
        )
        X=np.arange(1,35,0.1)
        Y=np.arange(1,35,0.1)
        X,Y=np.meshgrid(X,Y)
        Z=np.sqrt(1/(
            msVals["Eps"].iloc[0]+
            msVals["Eps1"].iloc[0]*1/effort+
            msVals["Eps2"].iloc[0]*interWorkoutFatigue/effort+
            msVals["Eps3"].iloc[0]*interExerciseFatigue/effort+
            msVals["Eps4"].iloc[0]*((X-1)**2)*((Y-1)**2)/effort+
            msVals["Eps5"].iloc[0]*((X-1)**2)/effort+
            msVals["Eps6"].iloc[0]*((Y-1)**2)/effort
        ))
        V=X*Y*Z
        V[V<-10]=np.nan
        maxIntensity=Z[0][0]
        fig=go.Figure(data=[go.Surface(x=X, y=Y, z=V,
            contours={
                "x": {"show": True, "start": 0, "end": 15},
                "y": {"show": True, "start": 0, "end": 15},
            },
        )])
        fig.update_layout(
            scene={
                "xaxis_title": "Sets {0}".format(Units.formatUnits(["Sets"])),
                "yaxis_title": "Reps {0}".format(Units.formatUnits(["Reps"])),
                "zaxis_title": "Volume {0}".format(Units.formatUnits(["PercentVolume"])),
                "zaxis": { "nticks": 12, "range": [0,200], },
            }, scene_camera={
                "up": { "x": 0, "y": 0, "z": 1 },
                "center": { "x": 0, "y": 0, "z": 0 },
                "eye": { "x": 1, "y": -1, "z": 1.5 },
            }, title={
                "text": "Volume From Basic Potential Surface for {0} at RPE {1}<br>"
                        "<sup>I(0,0)={2}  MSE: {3:0.2E}<br>"
                        "Date: {4}  Window: {5}  Time Frame: {6}</sup>"
                        .format(
                            liftName,
                            effort,
                            maxIntensity,
                            msVals["Mse"].iloc[0],
                            date,
                            msVals["Win"].iloc[0],
                            msVals["TimeFrame"].iloc[0],
                        ),
                "x": 0.5,
                "xanchor": "center",
            },
            width=IMAGE_SIZES["square"]["x"],
            height=IMAGE_SIZES["square"]["y"],
            margin={
                "b": 30,
            },
        )
        fig.write_image(
            "{0}/Volume/{1}.Effort{2}.volumeBase.png"
            .format(Ch3.fileLoc,liftName,effort)
        )

    @staticmethod
    @timerDecorator
    def eps1OverTime() -> None:
        tmp=Data.ms.query(
            "ClientID==1 and StateGeneratorID==1 and PotentialSurfaceID==1"
        )[["Eps1","Date"]].dropna()
        min: float=tmp["Eps1"].min()
        max: float=tmp["Eps1"].max()
        print((tmp["Eps1"]<0).sum())
        print((tmp["Eps1"]<=0).sum())
        numBelowZero: int=(tmp["Eps1"]<=0).sum()
        fig=px.scatter(tmp,x="Date",y="Eps1",
            labels={
                "Eps1": "$\\epsilon_1 \\text{{ {0} }}$".format(Units.formatUnits(["Constant"])),
                "Date": "Date {0}".format(Units.formatUnits(["Date"])),
            },
            title="$\\epsilon_1 "
                "\\text{{Over Time for Main Compound and Main Compound Accessories}}"
                "\\\\ \\epsilon_1\\in [{0:0.4E},{1:0.4E}]"
                "\\; \\;\\; |\\epsilon_1<=0|={2}"
                "\\; \\;\\; |\\epsilon_1|={3}$"
                .format(min,max,numBelowZero,len(tmp)),
            #log_y=True,
        )
        fig.update_layout(
            title={
                "x": 0.5,
                "xanchor": "center",
            },
            width=IMAGE_SIZES["thirdPage"]["x"],
            height=IMAGE_SIZES["thirdPage"]["y"],
        )
        fig.update_yaxes(range=[-0.1,0.1])
        fig.write_image("{0}/eps1OverTime.png".format(Ch3.fileLoc))

    @staticmethod
    @timerDecorator
    def volumeSkewOverTime() -> None:
        tmp=Data.vs.query(
            "ClientID==1 and StateGeneratorID==1 and PotentalSurfaceID==1"# and ExerciseID==15"
        )[["Date","VolumeSkew","ApproxVolumeSkew"]]#.dropna()
        # tmp["Date"]=pd.to_datetime(tmp["Date"])
        # numGte1: int=(tmp["VolumeSkew"]>=1).sum()
        # numLt1: int=(tmp["VolumeSkew"]<1).sum()
        # approxNumGte1: int=(tmp["ApproxVolumeSkew"]>=1).sum()
        # approxNumLt1: int=(tmp["ApproxVolumeSkew"]<1).sum()
        # print(numGte1,numLt1)
        fig=px.scatter(tmp,x="Date",y=tmp.columns[1:],
            labels={
                "VolumeSkew": "Volume Skew".format(Units.formatUnits(["Constant"])),
                "ApproxVolumeSkew": "Approximate Volume Skew".format(Units.formatUnits(["Constant"])),
                "Date": "Date {0}".format(Units.formatUnits(["Date"])),
            },
            title="Volume Skew Over Time",
            # title="Volume Skew Over Time<br>"
            #     "<sup>Volume Skew: >=1: {0}  <1: {1}<br>"
            #     "Approx Volume Skew: >=1: {2}  <1: {3}</sup>"
            #     .format(numGte1,numLt1,approxNumGte1,approxNumLt1)
        )
        print(np.average(np.abs(tmp["VolumeSkew"]-tmp["ApproxVolumeSkew"]).dropna()))
        # fig=px.scatter(x=tmp["Date"],y=np.abs(tmp["VolumeSkew"]-tmp["ApproxVolumeSkew"]))
        # fig.add_trace(go.Scatter(x=tmp["Date"],y=tmp["VolumeSkew"]-tmp["ApproxVolumeSkew"]))
        fig.update_layout(
            shapes=[{
                'type': 'line',
                'x0': tmp["Date"].min(),
                'x1': tmp["Date"].max(),
                'y0': 1,
                'y1': 1,
                'line': {
                    'color': 'rgb(50,171,96)',
                    'width': 3,
                },
            }],
            title={
                "x": 0.5,
                "xanchor": "center",
            },
            width=IMAGE_SIZES["thirdPage"]["x"],
            height=IMAGE_SIZES["thirdPage"]["y"],
        )
        fig.write_image("{0}/volumeSkewOverTime.png".format(Ch3.fileLoc))

    @staticmethod
    @timerDecorator
    def approxVolumeSkewGraphs():
        fig=make_subplots(cols=2,rows=2,
            vertical_spacing=0.2,
        )
        for i,f in enumerate(Data.approxVolSkew):
            col=i%2+1
            row=int(i/2)+1
            iterData=Data.approxVolSkew[f]
            fig.add_trace(go.Scatter(
                x=iterData[f],
                y=iterData["VolSkew"],
                name="Actual Volume Skew",
                legendgroup="actual",
                line={
                    "color": "red",
                },
                showlegend=(i==0),
            ),row=row, col=col)
            fig.add_trace(go.Scatter(
                x=iterData[f],
                y=iterData["ApproxVolSkew"],
                name="Approx. Volume Skew",
                legendgroup="approx",
                line={
                    "color": "blue",
                },
                showlegend=(i==0),
            ),row=row, col=col)
            fig.update_xaxes(
                title_text="{0} {1}".format(f,Units.formatUnits(["Constant"])),
                row=row,col=col,
            )
            if col==1:
                fig.update_yaxes(
                    title_text="Volume Skew {0}".format(Units.formatUnits(["Constant"])),
                    row=row,col=col,
                )
            fig.update_yaxes(
                range=[0,4],
                row=row,col=col,
            )
        fig.update_layout(
            width=IMAGE_SIZES["thirdPage"]["x"],
            height=IMAGE_SIZES["thirdPage"]["y"],
            title={
                "x": 0.5,
                "xanchor": "center",
            },
            title_text="Volume Skew vs Approx. Volume Skew Over Different Constant Ranges",
        )
        fig.write_image(
            "{0}/ApproxVsActualVolumeSkew.basic.png"
            .format(Ch3.fileLoc)
        )


if __name__=="__main__":
    #Ch3.eps1OverTime()
    #Ch3.volumeSkewOverTime()
    Ch3.allPlots("05/06/2023")
    #Ch3.approxVolumeSkewGraphs()
