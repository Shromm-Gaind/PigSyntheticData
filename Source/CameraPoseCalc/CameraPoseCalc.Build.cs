// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class CameraPoseCalc : ModuleRules
{
	public CameraPoseCalc(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "HeadMountedDisplay", "EnhancedInput","LevelSequence","Json","JsonUtilities" });
	}
}
