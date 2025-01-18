from collections import defaultdict

import numpy as np
import sqlalchemy
from InquirerPy.prompts import confirm
from sqlalchemy import engine
from sqlalchemy import orm

from backend import schema
from common import dataclassframe
from phase3 import collection
from phase3 import sim_config

SatStates = dict[str, dataclassframe.DataClassFrame[collection.State]]
TargetStates = dict[str, dataclassframe.DataClassFrame[collection.State]]
Measurements = dict[str, dataclassframe.DataClassFrame[collection.Measurement]]


class DbClient:
    DB_USER = 'simuser'
    DB_PASSWORD = 'simpassword'
    DB_HOST = 'localhost'
    DB_PORT = '5432'  # Port exposed by the container
    DB_NAME = 'satellite_sim'
    DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

    def __init__(self):
        self._engine = sqlalchemy.create_engine(self.DATABASE_URL)
        self.__session = orm.sessionmaker(bind=self._engine)()

        self._conn: engine.Connection | None = None

        # Tolerance for floating point comparisons
        self._epsilon = 1e-6

    @property
    def _session(self) -> orm.Session:
        if self._conn is None:
            try:
                self._conn = self._engine.connect()
            except Exception as e:
                raise Exception(f'Failed to connect to PostgreSQL: {e}')
            # Create the tables if they don't exist
            schema.Base.metadata.create_all(self._engine)

        return self.__session

    def connected(self) -> bool:
        return self._conn is not None

    def connect(self) -> bool:
        try:
            self._session
            return True
        except Exception as e:
            print(f'Failed to connect to PostgreSQL: {e}')
            return False

    def _get_or_create_scenario(self, cfg: sim_config.SimConfig) -> schema.Scenario:
        scenario = (
            self._session.query(schema.Scenario)
            .filter_by(name=cfg.name, rng_seed=cfg.rng_seed)
            .one_or_none()
        )

        if scenario is None:
            scenario = schema.Scenario(
                name=cfg.name,
                rng_seed=cfg.rng_seed,
                num_sensing_sats=len(cfg.sensing_satellites),
                num_fusion_sats=len(cfg.fusion_satellites),
                num_targets=sum(raid.initial_targs for raid in cfg.raids.values()),
            )
            self._session.add(scenario)
            self._session.commit()

        return scenario

    def _get_scenario(self, name: str, rng_seed: int) -> schema.Scenario:
        scenario = (
            self._session.query(schema.Scenario)
            .filter_by(name=name, rng_seed=rng_seed)
            .one_or_none()
        )

        if scenario is None:
            raise ValueError(f'Scenario "{name}" with RNG seed {rng_seed} not found.')

        return scenario

    def _sat_state_exists(
        self, scenario_id: sqlalchemy.Column[int], sat_name: str, time: float
    ) -> bool:
        return (
            self._session.query(schema.SatelliteState)
            .filter(
                sqlalchemy.and_(
                    schema.SatelliteState.scenario_id == scenario_id,
                    schema.SatelliteState.sat_name == sat_name,
                    schema.SatelliteState.time >= time - self._epsilon,
                    schema.SatelliteState.time <= time + self._epsilon,
                )
            )
            .count()
            > 0
        )

    def _target_state_exists(
        self, scenario_id: sqlalchemy.Column[int], target_id: str, time: float
    ) -> bool:
        return (
            self._session.query(schema.TargetState)
            .filter(
                sqlalchemy.and_(
                    schema.TargetState.scenario_id == scenario_id,
                    schema.TargetState.target_id == target_id,
                    schema.TargetState.time >= time - self._epsilon,
                    schema.TargetState.time <= time + self._epsilon,
                )
            )
            .count()
            > 0
        )

    def _measurement_exists(
        self, scenario_id: sqlalchemy.Column[int], target_id: str, time: float
    ) -> bool:
        return (
            self._session.query(schema.Measurement)
            .filter(
                sqlalchemy.and_(
                    schema.Measurement.scenario_id == scenario_id,
                    schema.Measurement.target_id == target_id,
                    schema.Measurement.time >= time - self._epsilon,
                    schema.Measurement.time <= time + self._epsilon,
                )
            )
            .count()
            > 0
        )

    def create_scenario(self, cfg: sim_config.SimConfig) -> None:
        """Create a scenario in the database.

        Args:
            cfg: The simulation configuration.
        """
        self._get_or_create_scenario(cfg)

    def insert_scenario(
        self,
        cfg: sim_config.SimConfig,
        sat_states: SatStates,
        target_states: TargetStates,
        measurements: Measurements,
    ):
        """Insert a scenario into the database.

        Args:
            cfg: The simulation configuration.
            sat_states: The states of the satellites.
            target_states: The states of the targets.
            measurements: The measurements.
        """
        scenario = self._get_or_create_scenario(cfg)
        scenario_id = scenario.id

        for sat_name, states in sat_states.items():
            states_dc = states.to_dataclasses()
            for state in states_dc:
                if self._sat_state_exists(scenario_id, sat_name, state.time):
                    continue
                state_schema = schema.SatelliteState(
                    scenario_id=scenario_id,
                    sat_name=sat_name,
                    time=state.time,
                    x=state.x,
                    y=state.y,
                    z=state.z,
                    vx=state.vx,
                    vy=state.vy,
                    vz=state.vz,
                )
                self._session.add(state_schema)

        for target_id, states in target_states.items():
            states_dc = states.to_dataclasses()
            for state in states_dc:
                if self._target_state_exists(scenario_id, target_id, state.time):
                    continue
                state_schema = schema.TargetState(
                    scenario_id=scenario_id,
                    target_id=target_id,
                    time=state.time,
                    x=state.x,
                    y=state.y,
                    z=state.z,
                    vx=state.vx,
                    vy=state.vy,
                    vz=state.vz,
                )
                self._session.add(state_schema)

        for measurements_df in measurements.values():
            measurements_dc = measurements_df.to_dataclasses()
            for measurement in measurements_dc:
                if self._measurement_exists(
                    scenario_id, measurement.target_id, measurement.time
                ):
                    continue
                measurement_schema = schema.Measurement(
                    scenario_id=scenario_id,
                    target_id=measurement.target_id,
                    time=measurement.time,
                    alpha=measurement.alpha,
                    beta=measurement.beta,
                    sat_name=measurement.sat_name,
                    sat_state=measurement.sat_state.tobytes(),
                    R_mat=measurement.R_mat.tobytes(),
                )
                self._session.add(measurement_schema)

        self._session.commit()

    def insert_time_step_states(
        self,
        name: str,
        sat_states: dict[str, collection.State],
        target_states: dict[str, collection.State],
        rng_seed: int = 1337,
    ):
        """Insert a time step into the database.

        Args:
            name: The name of the scenario.
            sat_states: A dictionary of satellite name -> state.
            target_state: A dictionary of target ID -> state
            rng_seed: The RNG seed of the scenario.
        """
        scenario = self._get_scenario(name, rng_seed)
        scenario_id = scenario.id

        for sat_name, state in sat_states.items():
            if self._sat_state_exists(scenario_id, sat_name, state.time):
                print(f'Satellite state unexpectedly exists in DB: {state}')
                continue
            state_schema = schema.SatelliteState(
                scenario_id=scenario_id,
                sat_name=sat_name,
                time=state.time,
                x=state.x,
                y=state.y,
                z=state.z,
                vx=state.vx,
                vy=state.vy,
                vz=state.vz,
            )
            self._session.add(state_schema)

        for target_id, state in target_states.items():
            if self._target_state_exists(scenario_id, target_id, state.time):
                print(f'Target state unexpectedly exists in DB: {state}')
                continue
            state_schema = schema.TargetState(
                scenario_id=scenario_id,
                target_id=target_id,
                time=state.time,
                x=state.x,
                y=state.y,
                z=state.z,
                vx=state.vx,
                vy=state.vy,
                vz=state.vz,
            )
            self._session.add(state_schema)

        self._session.commit()

    def insert_time_step_measurements(
        self,
        name: str,
        measurements: dict[str, list[collection.Measurement]],
        rng_seed: int = 1337,
    ):
        """Insert a time step into the database.

        Args:
            name: The name of the scenario.
            measurements: A dictionary of satellite name -> measurements.
            rng_seed: The RNG seed of the scenario.
        """
        scenario = self._get_scenario(name, rng_seed)
        scenario_id = scenario.id

        for sat_measurements in measurements.values():
            for measurement in sat_measurements:
                if self._measurement_exists(
                    scenario_id, measurement.target_id, measurement.time
                ):
                    print(f'Measurement unexpectedly exists in DB: {measurement}')
                    continue
                measurement_schema = schema.Measurement(
                    scenario_id=scenario_id,
                    target_id=measurement.target_id,
                    time=measurement.time,
                    alpha=measurement.alpha,
                    beta=measurement.beta,
                    sat_name=measurement.sat_name,
                    sat_state=measurement.sat_state.tobytes(),
                    R_mat=measurement.R_mat.tobytes(),
                )
                self._session.add(measurement_schema)

        self._session.commit()

    def load_time_step_states(
        self, name: str, time: float, rng_seed: int = 1337
    ) -> tuple[
        dict[str, collection.State],
        dict[str, collection.State],
    ]:
        """Load a time step from the database.

        Args:
            name: The name of the scenario.
            time: The time of the time step.
            rng_seed: The RNG seed of the scenario.

        Returns:
            A tuple containing the satellite states and target states.
        """
        scenario = self._get_scenario(name, rng_seed)
        scenario_id = scenario.id

        sat_states = (
            self._session.query(schema.SatelliteState)
            .filter(
                sqlalchemy.and_(
                    schema.SatelliteState.scenario_id == scenario_id,
                    schema.SatelliteState.time >= time - self._epsilon,
                    schema.SatelliteState.time <= time + self._epsilon,
                )
            )
            .all()
        )
        target_states = (
            self._session.query(schema.TargetState)
            .filter(
                sqlalchemy.and_(
                    schema.TargetState.scenario_id == scenario_id,
                    schema.TargetState.time >= time - self._epsilon,
                    schema.TargetState.time <= time + self._epsilon,
                )
            )
            .all()
        )

        sat_states_dc = {}
        for sat_state in sat_states:
            sat_states_dc[sat_state.sat_name] = collection.State(
                time=sat_state.time,  # type: ignore
                x=sat_state.x,  # type: ignore
                y=sat_state.y,  # type: ignore
                z=sat_state.z,  # type: ignore
                vx=sat_state.vx,  # type: ignore
                vy=sat_state.vy,  # type: ignore
                vz=sat_state.vz,  # type: ignore
            )

        target_states_dc = {}
        for target_state in target_states:
            target_states_dc[target_state.target_id] = collection.State(
                time=target_state.time,  # type: ignore
                x=target_state.x,  # type: ignore
                y=target_state.y,  # type: ignore
                z=target_state.z,  # type: ignore
                vx=target_state.vx,  # type: ignore
                vy=target_state.vy,  # type: ignore
                vz=target_state.vz,  # type: ignore
            )

        return (
            sat_states_dc,
            target_states_dc,
        )

    def load_time_step_measurements(
        self, name: str, time: float, rng_seed: int = 1337
    ) -> dict[str, list[collection.Measurement]]:
        """Load a time step from the database.

        Args:
            name: The name of the scenario.
            time: The time of the time step.
            rng_seed: The RNG seed of the scenario.

        Returns:
            A dictionary of satellite name -> measurements.
        """
        scenario = self._get_scenario(name, rng_seed)
        scenario_id = scenario.id

        measurements = (
            self._session.query(schema.Measurement)
            .filter(
                sqlalchemy.and_(
                    schema.Measurement.scenario_id == scenario_id,
                    schema.Measurement.time >= time - self._epsilon,
                    schema.Measurement.time <= time + self._epsilon,
                )
            )
            .all()
        )

        measurements_dc = defaultdict(list)
        for measurement in measurements:
            measurements_dc[measurement.sat_name].append(
                collection.Measurement(
                    target_id=measurement.target_id,  # type: ignore
                    time=measurement.time,  # type: ignore
                    alpha=measurement.alpha,  # type: ignore
                    beta=measurement.beta,  # type: ignore
                    sat_name=measurement.sat_name,  # type: ignore
                    sat_state=np.frombuffer(measurement.sat_state),  # type: ignore
                    R_mat=np.frombuffer(measurement.R_mat),  # type: ignore
                )
            )

        return measurements_dc

    def load_scenario(
        self, name: str, rng_seed: int = 1337
    ) -> tuple[SatStates, TargetStates, Measurements]:
        """Load a scenario from the database.

        TODO: This is super slow. Much slower than querying each time step.
        Look into the DataclassFrame usage.

        Args:
            name: The name of the scenario.
            rng_seed: The RNG seed of the scenario.

        Returns:
            A tuple containing the satellite states, target states, and measurements.
        """
        scenario = self._get_scenario(name, rng_seed)
        scenario_id = scenario.id

        sat_states = (
            self._session.query(schema.SatelliteState)
            .filter_by(scenario_id=scenario_id)
            .all()
        )
        target_states = (
            self._session.query(schema.TargetState)
            .filter_by(scenario_id=scenario_id)
            .all()
        )
        measurements = (
            self._session.query(schema.Measurement)
            .filter_by(scenario_id=scenario_id)
            .all()
        )

        sat_states_dc = defaultdict(
            lambda: dataclassframe.DataClassFrame(clz=collection.State)
        )
        for sat_state in sat_states:
            sat_states_dc[sat_state.sat_name].append(
                collection.State(
                    time=sat_state.time,  # type: ignore
                    x=sat_state.x,  # type: ignore
                    y=sat_state.y,  # type: ignore
                    z=sat_state.z,  # type: ignore
                    vx=sat_state.vx,  # type: ignore
                    vy=sat_state.vy,  # type: ignore
                    vz=sat_state.vz,  # type: ignore
                )
            )

        target_states_dc = defaultdict(
            lambda: dataclassframe.DataClassFrame(clz=collection.State)
        )
        for target_state in target_states:
            target_states_dc[target_state.target_id].append(
                collection.State(
                    time=target_state.time,  # type: ignore
                    x=target_state.x,  # type: ignore
                    y=target_state.y,  # type: ignore
                    z=target_state.z,  # type: ignore
                    vx=target_state.vx,  # type: ignore
                    vy=target_state.vy,  # type: ignore
                    vz=target_state.vz,  # type: ignore
                )
            )

        measurements_dc = defaultdict(
            lambda: dataclassframe.DataClassFrame(clz=collection.Measurement)
        )
        for measurement in measurements:
            measurements_dc[measurement.sat_name].append(
                collection.Measurement(
                    target_id=measurement.target_id,  # type: ignore
                    time=measurement.time,  # type: ignore
                    alpha=measurement.alpha,  # type: ignore
                    beta=measurement.beta,  # type: ignore
                    sat_name=measurement.sat_name,  # type: ignore
                    sat_state=np.frombuffer(measurement.sat_state),  # type: ignore
                    R_mat=np.frombuffer(measurement.R_mat),  # type: ignore
                )
            )

        return (
            sat_states_dc,
            target_states_dc,
            measurements_dc,
        )

    def remove_scenario(self, name: str, rng_seed: int = 1337) -> None:
        """Remove a scenario from the database.

        Args:
            name: The name of the scenario.
            rng_seed: The RNG seed of the scenario.
        """
        scenario = self._get_scenario(name, rng_seed)

        if not confirm.ConfirmPrompt(
            message=f'Are you sure you want to delete scenario "{name}" with RNG seed {rng_seed}?'
        ).execute():
            print('Aborting...')
            return

        # Delete the scenario (deletions will cascade to other tables)
        try:
            self._session.delete(scenario)
            self._session.commit()
        except Exception as e:
            print(f'Failed to remove scenario "{name}" with RNG seed {rng_seed}: {e}')
            print('Rolling back...')
            self._session.rollback()
            raise e
