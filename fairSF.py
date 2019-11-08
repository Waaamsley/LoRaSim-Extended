class fairSF():

    def __init__(self, nodeCount, sfList):
        self.node_count = nodeCount
        self.sf_list = sfList
        return


    def base_function(self):
        sum_result = 0.0

        for sf in self.sf_list:
            sum_result += sf/(2**sf)

        return sum_result


    def get_percentages(self):
        sf_percentages = []

        for sf in self.sf_list:
            sf_percentages.append(self.get_percentage(sf))

        return sf_percentages


    def get_percentage(self, sf):
        sf_percentage = 0.0

        sum_result = self.base_function()
        sf_percentage =  (sf/(2**sf)) * sum_result

        return sf_percentage

