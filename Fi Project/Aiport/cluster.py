"""
cluster algorithms
"""


class db_scan:
    def __init__(self, radius, density, distance):
        """
        :param radius: [float]radius of core point
        :param density: [float]density of core region
        :param distance: [float]distance function
        """
        self.radius, self.density, self.distance = radius, density, distance
        return
