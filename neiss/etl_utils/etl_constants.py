class feConstants:
    """
    Constants useful for feature extraction. If the names ever need to be changed, changes here will propagate.
    """
    DT_FMT = '%m/%d/%Y'
    MONTH = '%m'
    NOMINAL_FEATS = ('sex', 'race', 'stratum', 'diag', 'body_part', 'location')
    TO_ADD = ('month', 'age','disposition')

class colNames:
    """
    Constants in raw data files.
    """
    CASE_NO = 'CPSC Case #'
    TRMT_DATE = 'trmt_date'
    PSU = 'psu'
    WEIGHT = 'weight'
    STRATUM = 'stratum'
    AGE = 'age'
    SEX = 'sex'
    RACE = 'race'
    RACE_OTHER = 'race_other'
    DIAG = 'diag'
    DIAG_OTHER = 'diag_other'
    BODY_PART = 'body_part'
    DISPOSITION = 'disposition'
    LOCATION = 'location'
    FMV = 'fmv'
    PROD1 = 'prod1'
    PROD2 = 'prod2'
    NARR1 = 'narr1'
    NARR2 = 'narr2'

class featNames:
    """
    Extracted feature constants.
    """
    SEX_0 = 'sex_0'
    SEX_1 = 'sex_1'
    SEX_2 = 'sex_2'
    RACE_0 = 'race_0'
    RACE_1 = 'race_1'
    RACE_2 = 'race_2'
    RACE_3 = 'race_3'
    RACE_4 = 'race_4'
    RACE_5 = 'race_5'
    RACE_6 = 'race_6'
    STRATUM_C = 'stratum_C'
    STRATUM_L = 'stratum_L'
    STRATUM_M = 'stratum_M'
    STRATUM_S = 'stratum_S'
    STRATUM_V = 'stratum_V'
    DIAG_41 = 'diag_41'
    DIAG_42 = 'diag_42'
    DIAG_46 = 'diag_46'
    DIAG_47 = 'diag_47'
    DIAG_48 = 'diag_48'
    DIAG_49 = 'diag_49'
    DIAG_50 = 'diag_50'
    DIAG_51 = 'diag_51'
    DIAG_52 = 'diag_52'
    DIAG_53 = 'diag_53'
    DIAG_54 = 'diag_54'
    DIAG_55 = 'diag_55'
    DIAG_56 = 'diag_56'
    DIAG_57 = 'diag_57'
    DIAG_58 = 'diag_58'
    DIAG_59 = 'diag_59'
    DIAG_60 = 'diag_60'
    DIAG_61 = 'diag_61'
    DIAG_62 = 'diag_62'
    DIAG_63 = 'diag_63'
    DIAG_64 = 'diag_64'
    DIAG_65 = 'diag_65'
    DIAG_66 = 'diag_66'
    DIAG_67 = 'diag_67'
    DIAG_68 = 'diag_68'
    DIAG_69 = 'diag_69'
    DIAG_71 = 'diag_71'
    DIAG_72 = 'diag_72'
    DIAG_73 = 'diag_73'
    DIAG_74 = 'diag_74'
    BODY_PART_0 = 'body_part_0'
    BODY_PART_30 = 'body_part_30'
    BODY_PART_31 = 'body_part_31'
    BODY_PART_32 = 'body_part_32'
    BODY_PART_33 = 'body_part_33'
    BODY_PART_34 = 'body_part_34'
    BODY_PART_35 = 'body_part_35'
    BODY_PART_36 = 'body_part_36'
    BODY_PART_37 = 'body_part_37'
    BODY_PART_38 = 'body_part_38'
    BODY_PART_75 = 'body_part_75'
    BODY_PART_76 = 'body_part_76'
    BODY_PART_77 = 'body_part_77'
    BODY_PART_79 = 'body_part_79'
    BODY_PART_80 = 'body_part_80'
    BODY_PART_81 = 'body_part_81'
    BODY_PART_82 = 'body_part_82'
    BODY_PART_83 = 'body_part_83'
    BODY_PART_84 = 'body_part_84'
    BODY_PART_85 = 'body_part_85'
    BODY_PART_87 = 'body_part_87'
    BODY_PART_88 = 'body_part_88'
    BODY_PART_89 = 'body_part_89'
    BODY_PART_92 = 'body_part_92'
    BODY_PART_93 = 'body_part_93'
    BODY_PART_94 = 'body_part_94'
    LOCATION_0 = 'location_0'
    LOCATION_1 = 'location_1'
    LOCATION_2 = 'location_2'
    LOCATION_4 = 'location_4'
    LOCATION_5 = 'location_5'
    LOCATION_6 = 'location_6'
    LOCATION_7 = 'location_7'
    LOCATION_8 = 'location_8'
    LOCATION_9 = 'location_9'
    MONTH = 'month'
    AGE = 'age'
    DISPOSITION = 'disposition'
    TARG_1 = 'targ_1'
    TARG_2 = 'targ_2'
    TARG_4 = 'targ_4'
    TARG_5 = 'targ_5'
    TARG_6 = 'targ_6'
    TARG_8 = 'targ_8'
    TARG_9 = 'targ_9'