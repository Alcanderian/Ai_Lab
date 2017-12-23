"""
cluster algorithms
"""


class db_scan:
    def __init__(self, radius, density, distance):
        """
        :param radius: radius of core point 
        :param density: density of core region
        :param distance: distance function
        """
        self.radius, self.density, self.distance = radius, density, distance
        return
