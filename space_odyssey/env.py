from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.readwrite import BIFReader
import networkx as nx


OBS_VARS = {
    'AI_Test', 'Diagnosis', 'Temperature', 'O2_Level',
    'CO2_Level', 'Porthole', 'Alert_System'
}
DO_VARS =  {
    'HAL_Switch', 'Thermostat', 'O2_Generator',
    'CO2_Scrubber', 'Manoeuvre', 'Decompression'
}
HIDDEN_VARS = {'HAL', 'System_Age', 'Life_Support', 'Meteor_Shower', 'Alien_Attack'}
TARGET_VARS = {'Crew_Status'}
ACTION_POINTS = 5


class SpaceOdysseySimulator:
    model: DiscreteBayesianNetwork
    def __init__(
            self,
            model: DiscreteBayesianNetwork,
            initial_state: dict = None,
            seed: int = None,
    ):
        self.seed = seed
        # define parameters
        self.model = model
        self.obs_vars = OBS_VARS
        self.do_vars = DO_VARS
        self.hidden_vars = HIDDEN_VARS
        self.target_vars = TARGET_VARS
        self.start_action_points = ACTION_POINTS

        self.steps = 0
        # track available action points
        self.action_points = self.start_action_points
        # initialize state
        self.initial_state = initial_state
        self._state = self.model.simulate(n_samples=1, evidence=self.initial_state, seed=self.seed).iloc[0].to_dict()

    def reset(self):
        """
        Reset simulation state
        :return:
        """
        self.steps = 0
        self.action_points = self.start_action_points
        self._state = self.model.simulate(n_samples=1, evidence=self.initial_state, seed=self.seed).iloc[0].to_dict()

    def observe(self, variable):
        """
        Reveal state value of a specific variable
        :param variable: name from obs_vars set
        :return: observation state
        """
        print(f'*** Step {self.steps} ***')
        if variable not in self.obs_vars:
            print(f"We can't observe {variable} directly.")
            return None

        if self.action_points < 1:
            print(f"You have no action points left.")
            return None

        obs_state = self._state[variable]
        self.action_points -= 1
        self.steps += 1

        print(f'You send your crew member to check {variable}.')
        print(f'{variable} is reported to be in {obs_state} state.')
        print(f'You have {self.action_points} action points left.')
        print()
        return obs_state

    def act(self, variable, value):
        """
        Force state of a variable (intervention)
        :param variable: name of a variable from do_vars set
        :param value: desirable state value
        :return:
        """
        print(f'*** Step {self.steps} ***')
        if variable not in self.do_vars:
            print(f"We can't change {variable} directly.")
            return

        if self.action_points < 2:
            print(f"Not enough action points.")
            return

        # apply do operator
        # first, we should remove all descendants of the variable from the state
        ds = nx.descendants(self.model, variable)
        self._state = {key: value for key, value in self._state.items() if key not in ds.union({variable})}
        # then resample state values using state as evidence
        self._state = self.model.simulate(n_samples=1, do={variable: value}, evidence=self._state, seed=self.seed).iloc[0].to_dict()
        self.action_points -= 2

        print(f'You send your crew member to fix {variable}.')
        print(f'{variable} is now {value}.')
        print(f'You have {self.action_points} action points left.')
        print()

    def finish(self):
        """
        Reveal states of target variables
        :return:
        """
        print('*** Finishing Simulation ***')
        for tv in self.target_vars:
            print(f'{tv}: {self._state[tv]}')


if __name__ == '__main__':
    bn = BIFReader('spaceship.bif').get_model()

    # ini_state = {'O2_Level': 'low', 'Temperature': 'cold', 'Alert_System': 'warning', 'Diagnosis': 'nominal'}
    # ini_state = {'O2_Level': 'low', 'Alert_System': 'warning', 'Diagnosis': 'anomaly'}
    # ini_state = {'Temperature': 'hot', 'Alert_System': 'silent', 'Diagnosis': 'nominal'}
    # ini_state = {'AI_Test': 'pass', 'Temperature': 'hot', 'O2_Level': 'low', 'Alert_System': 'silent', 'Diagnosis': 'nominal'}
    ini_state = {'CO2_Level': 'high', 'Alert_System': 'silent', 'Porthole': 'danger'}
    env = SpaceOdysseySimulator(bn, initial_state=ini_state, seed=3232)

    env.observe('AI_Test')
    env.observe('Porthole')
    env.act('HAL_Switch', 'off')

    env.finish()
