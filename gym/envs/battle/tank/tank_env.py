import gym
from gym import error, spaces
from gym import utils
import numpy as np


class TankEnv(gym.Env, utils.EzPickle):
    def __int__(self, game_args, game_map, tanks):
        self._action_set = {"turnTo": 4, "fire": 4, "move": 1, "noop": 1}  # 所有操作的合集：10种
        self.action_space = spaces.Discrete(99999)  # 假设有5辆坦克
        self.observation_space = spaces.Discrete(19*19*4444)  # 所有参数压缩成一个标量: 敌方坦克(4种方向），我方坦克（4种方向），子弹的位置（4种方向）,空间类型（4种类型：树林/石块/空地/旗帜）

        self.game_args = game_args
        self.game_map = game_map
        self.tanks = tanks

        self.latest_state = None
        self.order_set = None

        self.round_count = 0

    def _step(self, action):
        self._forward_step(action)

        if self.latest_state is not None:
            new_observation = self._parse_state()
            reward = self._cal_reward()
            is_done = self._is_done()
            self.latest_state = None  # clean latest_state
            return new_observation, reward, is_done, None

    def _reset(self):
        if self.latest_state is not None:
            new_observation = self._parse_state()
            return new_observation

    def _render(self, mode='human', close=False):
        pass

    def _seed(self, seed=None):
        return []

    def _forward_step(self, action):
        order_set = []

        def value2order(v):
            if 0 <= v < 4:
                return "turnTo", (v+1)
            if 4 <= v < 8:
                return "fire", (v-4+1)
            if v == 8:
                return "move", 1
            if v == 9:
                return "noop", 1

            return "noop", 1

        for i, a in enumerate(action):
            individual_tank_action = action // pow(10, 5-1-i)
            op = value2order(individual_tank_action)

            order = {
                "tank_id": self.tanks[i],
                "order": op[0],
                "dir": op[1]
            }

            order_set.append(order)

        self.order_set = order_set

    def _parse_state(self):
        map_status = np.zeros((19,19))

        for x in range(19):
            for y in range(19):
                point_status = 0

                # tanks
                enemy_tank = 0
                our_tank = 0
                shell = 0
                space = 4  # out-space cell

                for tank in self.latest_state['tanks']:
                    tank_pos = tank['pos']
                    tank_dir = tank['dir']

                    if x == tank_pos["x"] and y == tank_pos["y"]:
                        if tank['id'] in self.tanks:
                            our_tank = tank_dir
                        else:
                            enemy_tank = tank_dir

                for shell_item in self.latest_state['shells']:
                    shell_pos = shell_item['pos']
                    shell_dir = shell_item['dir']

                    if x == shell_pos['x'] and y == shell_pos['y']:
                        shell = shell_dir

                for x_, row in enumerate(self.game_map):
                    for y_, item in enumerate(row):
                        if x_ == x and y_ == y:
                            # 0 means empty field, 1 means barrier, 2 means woods, 3 means flag.
                            space = item

                flag_pos = self.latest_state['flagPos']
                if flag_pos:
                    if x == ["x"] and y == flag_pos["y"]:
                        space = 3

                point_status = enemy_tank * 1000 + our_tank * 100 + shell * 10 + space

                map_status[x, y] = point_status

        return map_status

    def _count_tank(self):
        our_tank = 0
        enemy_tank = 0

        for tank in self.latest_state['tanks']:
            if tank['id'] in self.tanks:
                our_tank += 1
            else:
                enemy_tank += 1

        return our_tank, enemy_tank

    def _cal_reward(self):
        our_tank = 0
        enemy_tank = 0

        our_tank, enemy_tank = self._count_tank()

        our_flag = self.latest_state["yourFlagNo"]
        enemy_flag = self.latest_state["enemyFlagNo"]

        return our_tank + our_flag > enemy_tank + enemy_flag

    def _is_done(self):
        our_tank, enemy_tank = self._count_tank()

        if our_tank == 0 or enemy_tank == 0:
            return True

        if self.round_count == self.game_args["maxRound"]:
            return True

        return False