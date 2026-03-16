export interface Session {
  session_key: number;
  session_name: string;
  session_type: string;
  date_start: string;
  date_end: string | null;
  year: number;
  meeting_key: number;
  circuit_short_name: string;
  country_name: string;
  circuit_key?: number;
  gmt_offset?: string;
  location?: string;
}

export interface Meeting {
  meeting_key: number;
  meeting_name: string;
  meeting_official_name: string;
  country_name: string;
  country_code: string;
  circuit_short_name: string;
  year: number;
  date_start: string;
  gmt_offset?: string;
  meeting_code?: string;
  location?: string;
}

export interface Driver {
  driver_number: number;
  broadcast_name: string;
  full_name: string;
  name_acronym: string;
  team_name: string;
  team_colour: string;
  headshot_url: string | null;
  country_code?: string;
  session_key: number;
  meeting_key: number;
}

export interface Position {
  driver_number: number;
  position: number;
  date: string;
  session_key: number;
  meeting_key: number;
}

export interface CarData {
  driver_number: number;
  speed: number;
  rpm: number;
  n_gear: number;
  throttle: number;
  brake: number;
  drs: number;
  date: string;
  session_key: number;
  meeting_key: number;
}

export interface Location {
  driver_number: number;
  x: number;
  y: number;
  z: number;
  date: string;
  session_key: number;
  meeting_key: number;
}

export interface Stint {
  driver_number: number;
  stint_number: number;
  compound: string;
  tyre_age_at_start: number;
  lap_start: number;
  lap_end: number | null;
  session_key: number;
  meeting_key: number;
}

export interface RaceControlMessage {
  date: string;
  message: string;
  category: string;
  flag: string | null;
  scope: string | null;
  sector: number | null;
  driver_number: number | null;
  lap_number: number | null;
  session_key: number;
  meeting_key: number;
}

export interface DriverStanding {
  driver: Driver;
  position: number;
  carData: CarData | null;
  stint: Stint | null;
  location: Location | null;
}
