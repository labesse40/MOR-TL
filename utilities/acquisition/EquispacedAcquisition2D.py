import numpy as np
from copy import deepcopy

from .Acquisition2D import Acquisition2D
from .Shot2D import Source2D, SourceSet2D, Receiver2D, ReceiverSet2D, Shot2D


class EQUI2DAcquisition(Acquisition2D):
    """
    Define a 2D acquisition with: \
        n equispaced lines of equispaced sources and m equispaced lines of equispaced receivers
    The receiver set is the same for all shots
    """
    def __init__(self,
                 dt=None,
                 **kwargs):
        """
        Parameters
        -----------
            dt : float
                Timestep
            kwargs : keyword arguments for acquisition function
                startFirstSourceLine
                endFirstSourceLine
                startLastSourceLine
                endLastSourceLine
                startFirstReceiversLine
                endFirstReceiversLine
                startLastReceiversLine
                endLastReceiversLine
                numberOfSourceLines
                sourcesPerLine
                numberOfReceiverLines
                receiversPerLine
        """
        super().__init__(dt, **kwargs)
        self.type = "equispacedAcquisition"


    def acquisition(self,
                    startFirstSourceLine,
                    endFirstSourceLine,
                    startFirstReceiversLine,
                    endFirstReceiversLine,
                    startLastSourceLine=None,
                    endLastSourceLine=None,
                    startLastReceiversLine=None,
                    endLastReceiversLine=None,
                    numberOfSourceLines=1,
                    sourcesPerLine=1,
                    numberOfReceiverLines=1,
                    receiversPerLine=1,
                    **kwargs):
        """
        Set the shots configurations

        Parameters
        ----------
            startFirstSourceLine : list of len 2
                Coordinates of the first source of the first source line
            endFirstSourceLine : list of len 2
                Coordinates of the last source of the first source line
            startLastSourceLine : list of len 2
                Coordinates of the first source of the last source line
            endLastSourceLine : list of len 2
                Coordinates of the last source of the last source line
            startFirstReceiversLine : list of len 2
                Coordinates of the first receiver of the first receiver line
            endFirstReceiversLine : list of len 2
                Coordinates of the last receiver of the first receiver line
            startLastReceiversLine : list of len 2
                Coordinates of the first receiver of the last receiver line
            endLastReceiversLine : list of len 2
                Coordinates of the last receiver of the last receiver line
            numberOfSourceLines : int
                Number of source lines \
                Default is 1
            sourcesPerLine : int or list
                Number of sources per line \
                If int: same number for all source lines \
                Default is 1
            numberOfReceiverLines : int
                Number of receiver lines \
                Default is 1
            receiversPerLine : int or list
                Number of sources per line \
                If int: same number for all receiver lines \
                Default is 1
        """
        if numberOfSourceLines == 1:
            startLastSourceLine = startFirstSourceLine
            endLastSourceLine = endFirstSourceLine

        if numberOfReceiverLines == 1:
            startLastReceiversLine = startFirstReceiversLine
            endLastReceiversLine = endFirstReceiversLine

        # Set the start and end positions of all sources lines
        startSourcePosition = self.__generateListOfEquiPositions(startFirstSourceLine, startLastSourceLine, numberOfSourceLines)
        endSourcePosition = self.__generateListOfEquiPositions(endFirstSourceLine, endLastSourceLine, numberOfSourceLines)

        # Set the start and end positions of all receivers lines
        startReceiversPosition = self.__generateListOfEquiPositions(startFirstReceiversLine, startLastReceiversLine, numberOfReceiverLines)
        endReceiversPosition = self.__generateListOfEquiPositions(endFirstReceiversLine, endLastReceiversLine, numberOfReceiverLines)

        # Set the receiver set
        receiverSet = ReceiverSet2D()
        for n in range(numberOfReceiverLines):
            if isinstance(receiversPerLine, int):
                numberOfReceivers = receiversPerLine
            elif isinstance(receiversPerLine, list):
                assert len(numberOfReceivers) == numberOfReceiverLines
                numberOfReceivers = receiversPerLine[n]
            else:
                raise TypeError("The parameter `numberOfReceivers` can only be an integer or a list of integer numbers")

            xr, yr = self.__generateEquiPositionsWithinLine(startReceiversPosition[n], endReceiversPosition[n], numberOfReceivers)

            receiverSet_temp = ReceiverSet2D([Receiver2D(x, y) for x, y in list(zip(xr, yr))])
            receiverSet.appendSet(deepcopy(receiverSet_temp))

        # Define all sources positions
        xs = []
        ys = []
        for n in range(numberOfSourceLines):
            if isinstance(sourcesPerLine, int):
                numberOfSources = sourcesPerLine
            else:
                numberOfSources = sourcesPerLine[n]

            xst, yst = self.__generateEquiPositionsWithinLine(startSourcePosition[n], endSourcePosition[n], numberOfSources)

            for i in range(len(xst)):
                xs.append(xst[i])
                ys.append(yst[i])

        # Define all shots configuration
        # 1 source = 1 shot
        shots = []
        for i in range(len(xs)):
            sourceSet = SourceSet2D()

            shotId = f"{i+1:05d}"
            srcpos = [xs[i], ys[i]]
            sourceSet.append(Source2D(*srcpos))
            shot = Shot2D(sourceSet, receiverSet, shotId)

            shots.append(deepcopy(shot))

        self.shots = shots

    
    def __generateListOfEquiPositions(self, firstLinePosition, lastLinePosition, numberOfLines):
        """
        Generate a list of equispaced lines start or end positions

        Parameters
        -----------
            firstLinePosition : float
                Coordinates of the first line point
            lastLinePosition : float
                Coordinates of the last line point
            numberOfLines : int
                Number of equispaced lines

        Returns
        --------
            positions : list of list of float
                Equispaced coordinates as required
        """
        assert len(firstLinePosition) == len(lastLinePosition)

        positions = [[x, y] for x, y in zip(
            np.linspace(firstLinePosition[0], lastLinePosition[0], numberOfLines),
            np.linspace(firstLinePosition[1], lastLinePosition[1], numberOfLines),
            )]
        
        return positions    
    

    def __generateEquiPositionsWithinLine(self, startPosition, endPosition, numberOfPositions):
        """
        Generate the x, y equispaced coordinates within a line

        Parameters
        -----------
            startPosition : float
                Coordinates of the start position
            lastLinePosition : float
                Coordinates of the end position
            numberOfPositions : int
                Number of equispaced points on the line

        Returns
        --------
            x : list
                List of x coordinates
            y : list
                List of y coordinates
        """
        if startPosition == endPosition:
            numberOfPositions = 1

        x = np.linspace(startPosition[0], endPosition[0], numberOfPositions).tolist()
        y = np.linspace(startPosition[1], endPosition[1], numberOfPositions).tolist()
        
        return x, y
