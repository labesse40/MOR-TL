import numpy as np


class Shot2D:
    """Class representing a shot configuration : a SourceSet of 1 Source and a ReceiverSet

    Attributes
    ----------
        source :
            Source object
        receivers :
            ReceiverSet object
        flag :
            A flag to say if the shot configuration has been simulated
            "Undone", "In Progress", "Done"
        id : str
            Number identification
        mesh : Mesh object
            Mesh of the problem
    """
    def __init__(self,
                 sourceSet=None,
                 receiverSet=None,
                 shotId=None):
        """ Constructor of Shot

        Parameters
        ----------
            sourceSet : SourceSet, optional
                Set of Sources \
            receiverSet : ReceiverSet, optional
                Set of receivers
            shotId : str
                Number identification
        """
        if sourceSet is None:
            sourceSet = SourceSet2D()
        else:
            assert isinstance(sourceSet, SourceSet2D), "SourceSet instance expected for `sourceSet` argument"
        self.sources = sourceSet

        if receiverSet is None:
            receiverSet = ReceiverSet2D()
        else:
            assert isinstance(receiverSet, ReceiverSet2D), "ReceiverSet instance expected for `receiverSet` argument"
        self.receivers = receiverSet

        self.flag = "Undone"
        self.dt = None
        self.id = shotId
        self.mesh = None


    def __eq__(self, other):
        if isinstance(self, other.__class__):
            if self.sources == other.sources and self.receivers == other.receivers:
                return True
            
        return False


    def __repr__(self):
        return 'Source position : \n'+str(self.sources) +' \n\n' + 'Receivers positions : \n' + str(self.receivers) + '\n\n'


    def getSourceList(self):
        """
        Return the list of all sources in the Shot configuration

        Returns
        --------
            list of Source
                list of all the sources
        """
        return self.sources.getList()


    def getSourceCoords(self):
        """
        Return the list of all sources coordinates in the Shot configuration

        Returns
        --------
            list of list
                list of all the sources coordinates
        """
        return self.sources.getSourceCoords()


    def getReceiverList(self):
        """
        Return the list of all receivers in the Shot configuration

        Returns
        --------
            list of Receiver
                list of all the sources
        """
        return self.receivers.getList()


    def getReceiverCoords(self):
        """
        Return the list of all receivers coordinates in the Shot configuration

        Returns
        --------
            list of list
                list of all the receivers coordinates
        """
        return self.receivers.getReceiverCoords()


class ShotPoint2D:
    """
    Class defining the methods common to shot points (Source or Receiver)

    Attributes
    -----------
        coords : list of float
            Coordinates of the shot point
    """
    def __init__(self, x, y):
        """
        Parameters
        -----------
            x : str, int or float
                x coordinate
            y : str, int or float
                y coordinate
        """
        self.updatePosition(x, y)


    def __str__(self):
        return f'Position of Shot point : {self.coords}'


    def __repr__(self):
        return f'ShotPoint({self.coords[0]}, {self.coords[1]})'


    def __eq__(self, other):
        if isinstance(self, other.__class__):
            if self.coords == other.coords:
                return True
            
        return False


    def updateCoordinate(self, coord, value):
        """
        Update one of the coordinates

        Parameters
        -----------
            coord : int
                Which coordinate to update \
                Choices are 0, 1
            value : float or int
                New value
        """
        assert coord in (0, 1), "coord can only be 0 (x) or 1 (y)"
        assert isinstance(value, float) or isinstance(value, int)

        self.coords[coord] = value


    def getPosition(self):
        """
        Return the position coordinates

        Returns
        -----------
            list
                Coordinates    
        """
        return self.coords
    

    def updatePosition(self, x, y):
        """
        Update all the coordinates

        Parameters
        -----------
            coords : list or array of len 2
                New coordinates
        """
        assert all(str(c).replace(".", "", 1).isdigit() or isinstance(c, float) or isinstance(c, int) for c in (x,y)), "Only numeric values are accepted"

        self.coords = [float(c) for c in (x, y)]


    def x(self):
        """
        Get the x position

        Returns
        --------
            float
                X coordinate
        """
        return self.coords[0]


    def y(self):
        """
        Get the y position

        Returns
        --------
            float
                Y coordinate
        """
        return self.coords[1]


    def isinBounds(self, bounds):
        """
        Check if the receiver is in the bounds
        
        Parameters
        -----------
            bounds : list or array of len 4
                Bounds of format \
                (xmin, xmax, ymin, ymax)
        
        Returns
        --------
            bool
                True if receiver is in bounds, False otherwise
        """
        if (self.x() >= bounds[0] and self.x() <= bounds[1] \
        and self.y() >= bounds[2] and self.y() <= bounds[3]):

            return True
        else:
            return False


class Receiver2D(ShotPoint2D):
    """A class representing a receiver

    Attributes
    ----------
        coords :
            Coordinates of the receiver
    """
    def __init__(self, x, y):
        """Constructor for the receiver

        Parameters
        ----------
            pos : len 2 array-like
                Coordinates for the receiver
        """
        super().__init__(x, y)


    def __str__(self):
        return f'Position of Receiver : {self.coords}'


    def __str__(self):
        return f'Position of Receiver : {self.coords}'


    def __repr__(self):
        return f'Receiver({self.coords[0]}, {self.coords[1]})'


class Source2D(ShotPoint2D):
    """A class representing a point source

    Attributes
    ----------
        coords : list of float
            Coordinates of the source
    """
    def __init__(self, x, y):
        """Constructor for the point source

        Parameters
        ----------
            coords : list of float
                Coordinates for the point source
        """
        super().__init__(x, y)


    def __str__(self):
        return f'Position of Source : {self.coords}'


    def __repr__(self):
        return f'Source({self.coords[0]}, {self.coords[1]})'



class ShotPointSet2D:
    """
    Class defining methods for sets of shot points

    Attributes
    -----------
        list : list
            List of ShotPoint
        number : int
            Number of ShotPoint in the set
    """
    def __init__(self, shotPointList=None):
        """
        Parameters
        -----------
            shotPointList : list
                List of ShotPoint \
                Default is None
        """
        self.updateList(shotPointList)

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            if self.number == other.number:
                for sp1 in self.list:
                    if self.list.count(sp1) != other.list.count(sp1):
                        return False
                return True
            
        return False


    def getList(self):
        """
        Return the list of Shot points in the set

        Returns
        --------
            list
                List of Shot Points
        """
        return self.list
    

    def updateList(self, newList=None):
        """
        Update the full list with a new one

        Parameters
        -----------
            newList : list
                New shot points set \
                Default is empty list (reset)
        """
        if newList is None:
            self.list = []
        else:
            assert (isinstance(newList, list) or isinstance(newList, tuple))
            assert all(isinstance(sp, ShotPoint2D) for sp in newList), "`shotPointList` should only contain `ShotPoint2D` instances"

            self.list = list(newList)

        self.number = len(self.list)


    def append(self, shotPoint=None):
        """
        Append a new shot point to the set

        Parameters
        -----------
            shotPoint : ShotPoint2D
                Element to be added
        """
        if shotPoint is not None:
            assert isinstance(shotPoint, ShotPoint2D), "Can only add a `ShotPoint2D` object to the set"

            self.list.append(shotPoint)
            self.number += 1


    def appendSet(self, shotPointSet):
        """
        Append a list or a set of Shot Points to the existing one

        Parameters
        -----------
            shotPointSet : list or ShotPointSet
                Set of shot points to be added
        """
        if isinstance(shotPointSet, list):
            shotPointList = shotPointSet
        elif isinstance(shotPointSet, ShotPointSet2D):
            shotPointList = shotPointSet.getList()
        else:
            raise TypeError("Only Sets and list objects are acceptable")
        
        for shotPoint in shotPointList:
            self.append(shotPoint)
 

    
class ReceiverSet2D(ShotPointSet2D):
    """
    Class representing a set receiver

    Attributes
    ----------
        list : list
            List of Receivers
        number : int
            Number of Receivers
    """
    def __init__(self, receiverList=None):
        """Constructor for the receiver set

        Parameters
        ----------
            receiverList : list of Receiver
                List of Receiver
        """
        super().__init__(receiverList)


    def __repr__(self):
        """TODO: TEST THIS. Not sure it works"""
        """String representation of the receiver set"""
        if self.number >= 10:
            return str(self.list[0:4])[:-1] + '...' + '\n' + str(self.list[-4:])[1:]
        else:
            return str(self.list)


    def keepReceiversWithinBounds(self, bounds):
        """
        Filter the list to keep only the ones in the given bounds

        Parameters
        -----------
            bounds : list or array of len 4
                Bounds of format \
                (xmin, xmax, ymin, ymax)
        """
        newList = []

        for receiver in self.list:
            if receiver.isinBounds(bounds):
                newList.append(receiver)

        self.updateList(newList)


    def append(self, receiver):
        """
        Append a new receiver to the receiver set

        Parameters
        ----------
            receiver : Receiver
                Receiver to be added
        """
        assert isinstance(receiver, Receiver2D)
        super().append(receiver)


    def getReceiver(self, i):
        """
        Get a specific receiver from the set with its index

        Parameters
        -----------
            i : int
                Index of the receiver requested
        """
        if len(self.list) - 1 >= i:
            return self.list[i]
        else:
            raise IndexError("The receiver set is smaller than the index requested")


    def getReceiverCoords(self):
        """
        Get the coordinates of all the receivers

        Returns
        --------
            receiverCoords : list of list of floats
                List of all the receivers positions
        """
        receiverCoords = [receiver.coords for receiver in self.getList()]
        return receiverCoords



class SourceSet2D(ShotPointSet2D):
    """
    Class representing a source set

    Attributes
    ----------
        list :
            List of sources
        number :
            Number of sources
    """
    def __init__(self,
                 sourceList=None):
        """Constructor for the source set

        Parameters
        ----------
            sourceList : list of Source
                List of sources
        """
        super().__init__(sourceList)


    def __repr__(self):
        """TODO: TEST THIS. Not sure it works"""
        """String representation of the source set"""
        if self.number >=10:
            return str(self.list[0:4])[:-1] + '...' + '\n' + str(self.list[-4:])[1:]
        else:
            return str(self.list)


    def append(self, source):
        """
        Append a new source to the source set

        Parameters
        ----------
            source : Source
                Source to be added
        """
        assert isinstance(source, Source2D)
        super().append(source)


    def getSource(self, i):
        """
        Get a specific source from the set with its index

        Parameters
        -----------
            i : int
                Index of the source requested
        """
        if len(self.list) - 1 >= i:
            return self.list[i]
        else:
            raise IndexError("The source set is smaller than the index requested")


    def getSourceCoords(self):
        """
        Get the coordinates of all the sources

        Returns
        --------
            sourceCoords : list of list of float
                List of all the source positions
        """
        sourceCoords = [source.coords for source in self.getList()]
        return sourceCoords


    def getCenter(self):
        """
        Get the position of the center of the SourceSet
        
        Returns
        --------
            center : tuple or None
                Central position of the source set
        """
        center = None

        if self.number > 0:
            center = tuple(np.mean(np.array(self.getSourceCoords()), axis=0))
        
        return center