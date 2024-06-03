// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "ActorMoverComponent.generated.h"


UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class CAMERAPOSECALC_API UActorMoverComponent : public UActorComponent
{
	GENERATED_BODY()

public:	
	// Sets default values for this component's properties
	UActorMoverComponent();

	UFUNCTION(BlueprintCallable, Category = "Movement")
	void MoveActorToPose(int32 PoseIndex);

	UFUNCTION(BlueprintCallable, Category = "Movement")
	int32 GetPoseCount() const;


protected:
	// Called when the game starts
	virtual void BeginPlay() override;

private:	
	UPROPERTY(EditAnywhere, Category = "Configuration")
	FString JsonFilePath;

	TArray<FTransform> CameraPoses;

	bool ParseJSONFile(const FString& FilePath);
		
};
