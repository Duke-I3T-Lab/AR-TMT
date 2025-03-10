using UnityEngine;
using System.Collections.Generic;

[CreateAssetMenu(fileName = "StagePositionsData", menuName = "MyGame/StagePositionsData", order = 1)]
public class StagePositionsData : ScriptableObject
{
    public List<Vector3> stagePositions;
}