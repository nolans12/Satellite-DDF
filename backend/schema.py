"""Schemas for storing position and measurement data in a database.

NOTE: This needs to match the schemas in phase3/collection.py
"""

import numpy as np
import sqlalchemy
from sqlalchemy import orm
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext import declarative

Base = declarative.declarative_base()


class Scenario(Base):
    __tablename__ = 'scenarios'
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    name = sqlalchemy.Column(sqlalchemy.String, unique=True)
    rng_seed = sqlalchemy.Column(sqlalchemy.Integer)
    num_sensing_sats = sqlalchemy.Column(sqlalchemy.Integer)
    num_fusion_sats = sqlalchemy.Column(sqlalchemy.Integer)
    num_targets = sqlalchemy.Column(sqlalchemy.Integer)

    # Relationship to other tables
    satellite_states = orm.relationship(
        'SatelliteState', back_populates='scenario', cascade='all, delete-orphan'
    )
    target_states = orm.relationship(
        'TargetState', back_populates='scenario', cascade='all, delete-orphan'
    )
    measurements = orm.relationship(
        'Measurement', back_populates='scenario', cascade='all, delete-orphan'
    )


class SatelliteState(Base):
    __tablename__ = 'satellite_states'
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    scenario_id = sqlalchemy.Column(
        sqlalchemy.Integer, sqlalchemy.ForeignKey('scenarios.id', ondelete='CASCADE')
    )
    sat_name = sqlalchemy.Column(sqlalchemy.String)
    time = sqlalchemy.Column(sqlalchemy.Float)
    x = sqlalchemy.Column(sqlalchemy.Float)
    y = sqlalchemy.Column(sqlalchemy.Float)
    z = sqlalchemy.Column(sqlalchemy.Float)
    vx = sqlalchemy.Column(sqlalchemy.Float)
    vy = sqlalchemy.Column(sqlalchemy.Float)
    vz = sqlalchemy.Column(sqlalchemy.Float)

    # Accelerate queries by indexing the scenario_id and time columns
    __table_args__ = (sqlalchemy.Index('ix_sat_scenario_time', 'scenario_id', 'time'),)

    # Relationship to the scenario table
    scenario = orm.relationship('Scenario', back_populates='satellite_states')


class TargetState(Base):
    __tablename__ = 'target_states'
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    scenario_id = sqlalchemy.Column(
        sqlalchemy.Integer, sqlalchemy.ForeignKey('scenarios.id', ondelete='CASCADE')
    )
    target_id = sqlalchemy.Column(sqlalchemy.String)
    time = sqlalchemy.Column(sqlalchemy.Float)
    x = sqlalchemy.Column(sqlalchemy.Float)
    y = sqlalchemy.Column(sqlalchemy.Float)
    z = sqlalchemy.Column(sqlalchemy.Float)
    vx = sqlalchemy.Column(sqlalchemy.Float)
    vy = sqlalchemy.Column(sqlalchemy.Float)
    vz = sqlalchemy.Column(sqlalchemy.Float)

    # Accelerate queries by indexing the scenario_id and time columns
    __table_args__ = (
        sqlalchemy.Index('ix_target_scenario_time', 'scenario_id', 'time'),
    )

    # Relationship to the scenario table
    scenario = orm.relationship('Scenario', back_populates='target_states')


class Measurement(Base):
    __tablename__ = 'measurements'
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    scenario_id = sqlalchemy.Column(
        sqlalchemy.Integer, sqlalchemy.ForeignKey('scenarios.id', ondelete='CASCADE')
    )
    target_id = sqlalchemy.Column(sqlalchemy.String)
    time = sqlalchemy.Column(sqlalchemy.Float)
    alpha = sqlalchemy.Column(sqlalchemy.Float)
    beta = sqlalchemy.Column(sqlalchemy.Float)
    sat_name = sqlalchemy.Column(sqlalchemy.String)
    sat_state = sqlalchemy.Column(postgresql.BYTEA)  # Serialized array
    R_mat = sqlalchemy.Column(postgresql.BYTEA)  # Serialized matrix

    # Accelerate queries by indexing the scenario_id and time columns
    __table_args__ = (
        sqlalchemy.Index('ix_measurement_scenario_time', 'scenario_id', 'time'),
    )

    # Relationship to the scenario table
    scenario = orm.relationship('Scenario', back_populates='measurements')
