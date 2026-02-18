
For save the road mapped 2D:
```bash
ros2 run nav2_map_server map_saver_cli -f mapV1uncertainty --ros-args -p save_map_timeout:=20000 -p map_subscribe_transient_local:=false -r map:=/planner_uncertainty
```