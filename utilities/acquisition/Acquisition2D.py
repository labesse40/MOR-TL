from copy import deepcopy
import numpy as np

from .Shot2D import Source2D, SourceSet2D, Receiver2D, ReceiverSet2D, Shot2D


class Acquisition2D:
    def __init__(self,
                 dt=None,
                 **kwargs):
        """
        Parameters
        ----------
            dt : float
                Timestep
            kwargs : keyword arguments
                sources : list of list of float
                    List of all sources coordinates
                    If None, error is raised
                receivers : list of list of float
                    List of all receivers coordinates
                    If None, error is raised
                acqId : int
                    Acquisition id \
                    Default is 1
        """
        self.type = "acquisition"

        self.acquisition(**kwargs)

        acqId = kwargs.get("acqId", 1)
        self.id = f"{acqId:05d}"

        self.dt = dt
        for shot in self.shots:
            if dt is not None:
                shot.dt = dt

   
    def acquisition(self,
                    sources=None,
                    receivers=None,
                    **kwargs):
        """
        Set the shots configurations
        The same set of receivers is used for all shots

        Please provide the same type of variable for `sources` and `receivers`

        Parameters
        -----------
            sources : list of list of float or str
            Sources coordinates \
            If `sources` is str, filename containing all sources coordinates
            receivers : list of list of float or str
            Receivers coordinates \
            If `receivers` is str, filename containing all receivers coordinates

        Examples
        ---------

            >>> srcList = [[1, 2], [4, 5]]
            >>> rcvList = [[7, 8], [10, 11], [13, 14]]
            >>> acquisition = Acquisition2D(sources=srcList, receivers=rcvList)

            >>> srcArr = np.array(srcList)
            >>> rcvArr = np.array(rcvList)
            >>> acquisition = Acquisition2D(sources=srcArr, receivers=rcvArr)

            >>> srcTxt = "sources.txt"
            >>> rcvTxt = "receivers.txt"
            >>> acquisition = Acquisition2D(sources=srcTxt, receivers=rcvTxt)
        """
        if sources is None or receivers is None:
            raise ValueError("Sources and receivers must be provided.")
        elif isinstance(sources, str) and isinstance(receivers, str):
            sources = np.loadtxt(sources)
            receivers = np.loadtxt(receivers)

        numberOfSources = len(sources)

        receiverSet = self.createreceiverset(receivers)

        shots = []

        for i in range(numberOfSources):
            sourceSet = SourceSet2D()  #1 source per shot
            shot_id = f"{i+1:05d}"
            sourceSet.append(Source2D(*sources[i]))

            shot = Shot2D(sourceSet, receiverSet, shot_id)
            shots.append(deepcopy(shot))

        self.shots = shots
    

    def createreceiverset(self, receivers):
        """
        Helper method to create a ReceiverSet from a list of receiver coordinates.

        Parameters
        ----------
        receivers : list of list of float
            List of receiver coordinates.

        Returns
        -------
        ReceiverSet
            A set of receivers created from the provided coordinates.
        """
        receiver_list = [Receiver2D(*receiver_coords) for receiver_coords in receivers]
        return ReceiverSet2D(receiver_list)


    def getSourceCenter(self):
        """
        Return the central position of the all the sources contained in the acquisition

        Returns
        -------
            2d list : Coordinates of the center
        """
        sourceSet = SourceSet2D()
        for shot in self.shots:
            sourceSet.appendSet(shot.getSourceList())

        center = sourceSet.getCenter()
        return center


    def splitAcquisition(self):
        """
        Split the shots such that one Acquisition = 1 Shot

        Returns
        --------
            listOfAcquisition : list
                list of Acquisition objects such that 1 Shot = 1 Acquisition
        """
        listOfAcquisition = []
        for shot in self.shots:
            a = Acquisition2D(sources=shot.getSourceCoords(), receivers=shot.getReceiverCoords(), dt=shot.dt)
            a.shots[0].id = shot.id
            a.shots[0].dt = shot.dt

            listOfAcquisition.append(a)

        return listOfAcquisition
