# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd
import numpy as np

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import simulate
from activitysim.core import inject
from activitysim.core import expressions

from .util import estimation

logger = logging.getLogger(__name__)


@inject.step()
def parking_location_choice_at_university(
        trips, tours, land_use,
        chunk_size, trace_hh_id):
    """
    This model selects a parking location for groups of trips that are on university campuses where
    the tour mode is auto.  Parking locations are sampled weighted by the number of parking spots.

    The main interface to this model is the parking_location_choice_at_university() function.
    This function is registered as an orca step in the example Pipeline.
    """

    trace_label = 'parking_location_choice_at_university'
    model_settings_file_name = 'parking_location_choice_at_university.yaml'
    model_settings = config.read_model_settings(model_settings_file_name)

    univ_codes_col = model_settings['LANDUSE_UNIV_CODE_COL_NAME']
    univ_codes = model_settings['UNIV_CODES_THAT_REQUIRE_PARKING']
    random_state = model_settings['RANDOM_STATE']

    parking_spaces_col = model_settings['LANDUSE_PARKING_SPACES_COL_NAME']
    parking_univ_code_col = model_settings['LANDUSE_PARKING_UNIV_CODE_COL_NAME']

    parking_tour_modes = model_settings['TOUR_MODES_THAT_REQUIRE_PARKING']

    trips = trips.to_frame()
    tours = tours.to_frame()
    land_use_df = land_use.to_frame()

    # initialize univ parking columns
    trips['parked_at_university'] = False
    tours['univ_parking_zone_id'] = pd.NA

    all_univ_zones = land_use_df[land_use_df[univ_codes_col].isin(univ_codes)].index

    # grabbing all trips and tours that have a destination on a campus and selected tour mode
    trip_choosers = trips[trips['destination'].isin(all_univ_zones)]
    print(trip_choosers.destination.value_counts())
    tour_choosers = tours[
        tours.index.isin(trip_choosers['tour_id'])
        & tours.tour_mode.isin(parking_tour_modes)]
    print(tour_choosers)

    # removing trips that did not have the right tour mode.  (Faster than merging tour mode first?)
    trip_choosers = trip_choosers[trip_choosers.tour_id.isin(tour_choosers.index)]
    trip_choosers.loc[trip_choosers['purpose'] != 'Home', 'parked_at_university'] = True

    logger.info("Running %s for %d tours", trace_label, len(tour_choosers))

    # Set parking locations for each university independently
    for univ_code in univ_codes:
        # selecting land use data
        univ_zones = land_use_df[land_use_df[univ_codes_col] == univ_code].reset_index()
        parking_univ_zones = land_use_df[land_use_df[parking_univ_code_col] == univ_code].reset_index()

        if len(univ_zones) == 0:
            logger.info("No zones found for university code: %s", univ_code)
            continue

        if (len(parking_univ_zones) == 0) or (parking_univ_zones[parking_spaces_col].sum() == 0):
            logger.info("No parking found for university code: %s", univ_code)
            continue

        # selecting tours that have trips attending this university's zone(s)
        univ_trip_choosers = trip_choosers[trip_choosers['destination'].isin(univ_zones.zone_id)]
        parking_tours = (tour_choosers.index.isin(univ_trip_choosers.tour_id))
        num_parking_tours = parking_tours.sum()

        # parking location is sampled based on the number of parking spaces
        tour_choosers.loc[parking_tours, 'univ_parking_zone_id'] = parking_univ_zones.zone_id.sample(
            n=num_parking_tours,
            weights=parking_univ_zones[parking_spaces_col],
            replace=True,
            random_state=random_state).to_numpy()

        logger.info("Selected parking locations for %s tours for university with code: %s",
                    num_parking_tours, univ_code)

    # Overriding school_zone_id in persons table
    trips.loc[trips.index.isin(trip_choosers.index),
              'parked_at_university'] = trip_choosers['parked_at_university']
    tours.loc[tours.index.isin(tour_choosers.index),
              'univ_parking_zone_id'] = tour_choosers['univ_parking_zone_id']


    pipeline.replace_table("trips", trips)
    pipeline.replace_table("tours", tours)

    tracing.print_summary('parking_location_choice_at_university zones',
                          tours['univ_parking_zone_id'],
                          value_counts=True)

    if trace_hh_id:
        tracing.trace_df(tours,
                         label=trace_label,
                         warn_if_empty=True)
